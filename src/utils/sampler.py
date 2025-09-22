import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from baal.bayesian.dropout import MCDropoutModule
import torch.nn.functional as F
from src.utils import compute_acc

class Active_Learning_Sampler():
    def __init__(
        self,
        device,
        train_set,
        path_to_save_indices: str,
        path_to_initial10: str,
        budget_ratio: float,
        method: str,
    ) -> None:
        self.device = device
        self.path_to_save_indices = path_to_save_indices
        self.path_to_initial10 = path_to_initial10
        self.method = method
        self.train_set = train_set
        self.num_total = len(train_set)
        self.budget = int(budget_ratio * self.num_total)
        self.indices = np.arange(self.num_total, dtype=np.int64)
        self.labeled_idxs = None
        self.unlabeled_idxs = None
        self.labeled_pool = None
        self.unlabeled_pool = None
        self.selected = None
        self.scores = None
        self.num_labeled = int(0.1 * self.num_total)
        self.stage = 0
    
    def _subset_to_dataframe(self, subset):
        ds = subset
        idx = list(ds.indices)
        base = ds.dataset
        while isinstance(base, torch.utils.data.Subset):
            idx = [int(base.indices[i]) for i in idx]
            base = base.dataset
        return base.df.iloc[idx].copy()

    def _save_labeled_df(self, labeled_pool, scores, stage: int):
        save_dir = Path(self.path_to_save_indices)
        save_dir.mkdir(parents=True, exist_ok=True)
        df = self._subset_to_dataframe(labeled_pool)
        if df.index.name is None:
            df.index.name = "index"

        score_col = scores.rename("score")
        df = df.join(score_col, how="left")
        
        df.to_csv(save_dir / f"labeled_pool_stage{stage}.csv", index=True)

    def initial(self):
        self.labeled_idxs = pd.read_csv(self.path_to_initial10)['index'].to_numpy(dtype=np.int64)
        self.selected = self.labeled_idxs

        self.unlabeled_idxs = self.indices[~np.isin(self.indices, self.labeled_idxs)]

        labeled_pool   = torch.utils.data.Subset(self.train_set, self.labeled_idxs)
        unlabeled_pool = torch.utils.data.Subset(self.train_set, self.unlabeled_idxs)

        s = np.ones(len(self.selected))
        scores = pd.Series(s, index=self.selected, dtype="float32")

        self._save_labeled_df(labeled_pool, scores, self.stage)

        return labeled_pool, unlabeled_pool
    
    def initial_make(self):
        self.labeled_idxs = np.random.choice(self.indices, size=self.num_labeled, replace=False)
        self.selected = self.labeled_idxs

        self.unlabeled_idxs = self.indices[~np.isin(self.indices, self.labeled_idxs)]

        labeled_pool   = torch.utils.data.Subset(self.train_set, self.labeled_idxs)
        unlabeled_pool = torch.utils.data.Subset(self.train_set, self.unlabeled_idxs)

        s = np.ones(len(self.selected))
        scores = pd.Series(s, index=self.selected, dtype="float32")

        self._save_labeled_df(labeled_pool, scores, self.stage)

        return labeled_pool, unlabeled_pool

    def sampling(self, model = None, train_loader = None, unlabeled_loader = None, lp_model = None, decoder = None):

        if self.method == "random":
            S = torch.rand(len(self.unlabeled_idxs), dtype=torch.float32, device=self.device)
            
        elif self.method == "entropy":
            model.eval()
            ent=[]
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    outputs=model(inputs)
                    p=torch.softmax(outputs, dim=-1)
                    H=torch.logsumexp(outputs, dim=-1)-(p*outputs).sum(dim=-1)
                    mask = (torch.arange(outputs.size(1), device=outputs.device)[None, :] < input_lengths[:, None])  # [B,T]
                    H = (H * mask).sum(dim=1) / input_lengths
                    ent.append(H)
            S=torch.cat(ent)

        elif self.method == "margin":
            model.eval(); margins=[]
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    outputs=model(inputs)
                    p = torch.softmax(outputs, dim=-1)
                    v, _ = torch.topk(p, k=2, dim=-1)
                    mask = (torch.arange(outputs.size(1), device=outputs.device)[None, :] < input_lengths[:, None])  # [B,T]
                    m = ((v[...,0] - v[...,1]) * mask).sum(dim=1) / input_lengths  # [B]
                    margins.append(-m)
            S = torch.cat(margins)

        elif self.method == "coreset":
            exit()
        
        elif self.method == 'loss_prediction':
            model.eval(); lp_model.eval()
            l_list = []
            mlp_outs = {}
            def _hook(_m, inputs,):
                x_to_decoder = inputs[0]
                mlp_outs["x"] = x_to_decoder.detach()
            h = model.decoder.register_forward_pre_hook(_hook)
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    outs = model(inputs)
                    l = lp_model(mlp_outs['x'])
                    l /= input_lengths.to(l.dtype).unsqueeze(-1)
                    l_list.append(l.squeeze(-1))
            h.remove()
            S = torch.cat(l_list)
        
        elif self.method == 'mcdropout':
            model.eval()
            mc_model = MCDropoutModule(model)
            T = 5
            scores_all = []
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    
                    base_logits = model(inputs)
                    base_probs  = F.softmax(base_logits, dim=-1)
                    B = base_probs.size(0)

                    base_preds = []
                    for b in range(B):
                        seq = base_probs[b, : input_lengths[b].item()].detach().cpu().numpy()
                        pred = decoder.greedy_decode(seq)
                        base_preds.append("".join(pred))

                    # --- 2) MC Dropout T회 예측 (각 샘플마다 T개의 문자열) ---
                    mc_per_sample = [[] for _ in range(B)]
                    for _ in range(T):
                        logits_t = mc_model(inputs)
                        probs_t  = F.softmax(logits_t, dim=-1)
                        for b in range(B):
                            seq_t = probs_t[b, : input_lengths[b].item()].detach().cpu().numpy()
                            pred_t = decoder.greedy_decode(seq_t)
                            mc_per_sample[b].append("".join(pred_t))

                    # --- 3) 샘플별 불확실도 스코어 산출 ---
                    # 기준 아이디어: 베이스 예측과 T회 예측의 평균 일치도 → 불확실도 = 100 - 평균일치도
                    batch_scores = []
                    for b in range(B):
                        acc_b = compute_acc([base_preds[b]] * T, mc_per_sample[b])  # % 단위
                        batch_scores.append(100.0 - acc_b)
                    scores_all.append(torch.tensor(batch_scores, dtype=torch.float32))
        
            S = torch.cat(scores_all)

        elif self.method == 'mcdropout_beam':
            model.eval()
            mc_model = MCDropoutModule(model)
            T = 5
            scores_all = []
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    
                    base_logits = model(inputs)
                    base_probs  = F.softmax(base_logits, dim=-1)
                    B = base_probs.size(0)

                    base_preds = []
                    for b in range(B):
                        seq = base_probs[b, : input_lengths[b].item()].detach().cpu().numpy()
                        pred = decoder.beam_decode(seq, beam_size=5)
                        base_preds.append("".join(pred))

                    mc_per_sample = [[] for _ in range(B)]
                    for _ in range(T):
                        logits_t = mc_model(inputs)
                        probs_t  = F.softmax(logits_t, dim=-1)
                        for b in range(B):
                            seq_t = probs_t[b, : input_lengths[b].item()].detach().cpu().numpy()
                            pred_t = decoder.beam_decode(seq_t, beam_size=5)
                            mc_per_sample[b].append("".join(pred_t))

                    batch_scores = []
                    for b in range(B):
                        acc_b = compute_acc([base_preds[b]] * T, mc_per_sample[b])  # % 단위
                        batch_scores.append(100.0 - acc_b)
                    scores_all.append(torch.tensor(batch_scores, dtype=torch.float32))
        
            S = torch.cat(scores_all)       
                
        elif self.method == "entropy_max":
            model.eval()
            ent=[]
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    outputs=model(inputs)
                    p=torch.softmax(outputs, dim=-1)
                    H=torch.logsumexp(outputs, dim=-1)-(p*outputs).sum(dim=-1)
                    mask = (torch.arange(outputs.size(1), device=outputs.device)[None, :] < input_lengths[:, None])  # [B,T]
                    neg_inf = torch.finfo(H.dtype).min
                    H_max = H.masked_fill(~mask, neg_inf).max(dim=1).values
                    ent.append(H_max)
            S=torch.cat(ent)
        
        elif self.method == 'mcdropout_xlogx':
            model.eval()
            mc_model = MCDropoutModule(model)
            T = 5
            scores_all = []
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    
                    base_logits = model(inputs)
                    base_probs  = F.softmax(base_logits, dim=-1)
                    B = base_probs.size(0)

                    base_preds = []
                    for b in range(B):
                        seq = base_probs[b, : input_lengths[b].item()].detach().cpu().numpy()
                        pred = decoder.greedy_decode(seq)
                        base_preds.append("".join(pred))

                    mc_per_sample = [[] for _ in range(B)]
                    for _ in range(T):
                        logits_t = mc_model(inputs)
                        probs_t  = F.softmax(logits_t, dim=-1)
                        for b in range(B):
                            seq_t = probs_t[b, : input_lengths[b].item()].detach().cpu().numpy()
                            pred_t = decoder.greedy_decode(seq_t)
                            mc_per_sample[b].append("".join(pred_t))

                    batch_scores = []
                    for b in range(B):
                        acc_b = compute_acc([base_preds[b]] * T, mc_per_sample[b])  # % 단위
                        batch_scores.append(100.0 - acc_b)
                    scores_all.append(torch.tensor(batch_scores, dtype=torch.float32))
        
            cer = torch.cat(scores_all)
            cer = (cer - cer.min()) / (cer.max() - cer.min()).clamp_min(1e-12)   # min-max scaling → [0,1]
            S = -(cer.clamp_min(1e-12)) * torch.log(cer.clamp_min(1e-12))
        
        elif self.method == "TPC":
            model.eval(); all_scores = []; axis_weight = torch.tensor([1.0, 1.0, 1.0], device=self.device)
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    B, T, J, C = inputs.shape

                    w = axis_weight.view(*([1]* (inputs.dim()-1)), -1)
                    x = inputs * w

                    t = torch.arange(T, device=self.device)
                    valid = t[None, :] < input_lengths[:, None]
                    valid_d = valid[:, 1:] & valid[:, :-1]

                    diffs = x[:, 1:, :, :] - x[:, :-1, :, :]

                    per_step = torch.linalg.norm(diffs, dim=-1)

                    per_step = per_step * valid_d[..., None]  # [B, T-1, J]

                    sums = per_step.sum(dim=(1, 2))
                    counts = (valid_d.sum(dim=1).clamp_min(1) * J)
                    seq_scores = sums / counts

                    all_scores.append(seq_scores.detach().cpu())
            
            S = torch.cat(all_scores, dim=0).detach().cpu()
        
        elif self.method == "SE":
            model.eval()
            ent=[]
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    outputs=model(inputs)
                    probs = F.softmax(outputs, dim=-1)
                    for i in range(probs.shape[0]):
                        seqs_str, log_scores = decoder.beam_probs(
                            probs[i, :input_lengths[i].item()].detach().cpu().numpy(), beam_size=5
                        )
                        logw  = log_scores - torch.logsumexp(log_scores, dim=0)
                        w = torch.exp(logw)
                        H = -(w * logw).sum().unsqueeze(0)
                        ent.append(H)
            S=torch.cat(ent).detach().cpu()

        elif self.method == "lc":
            model.eval()
            scores_all=[]
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = (
                        inputs.to(self.device),
                        input_lengths.to(self.device),
                    )
                    outputs=model(inputs)
                    probs = F.softmax(outputs, dim=-1)
                    for i in range(probs.shape[0]):
                        seqs_str, log_scores = decoder.beam_probs(
                            probs[i, :input_lengths[i].item()].detach().cpu().numpy(), beam_size=5
                        )
                        lens = torch.tensor([max(1, len(s)) for s in seqs_str], dtype=torch.float32)
                        best_idx = torch.argmax(log_scores)
                        best_len = lens[best_idx].clamp_min(1.0)
                        best = torch.exp(log_scores[best_idx] / best_len)
                        c = 1.0 - best
                        scores_all.append(c.unsqueeze(0))

            S=torch.cat(scores_all).detach().cpu()

        elif self.method == "tf-idf":
            from collections import Counter
            import math

            model.eval()
            labeled_df = self._subset_to_dataframe(torch.utils.data.Subset(self.train_set, self.labeled_idxs))
            labeled_docs = [str(s) for s in labeled_df["phrase"].tolist()]

            pred_docs = []
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = inputs.to(self.device), input_lengths.to(self.device)

                    outputs = model(inputs)
                    probs = F.softmax(outputs, dim=-1)

                    B = probs.shape[0]
                    for i in range(B):
                        T = int(input_lengths[i].item())
                        seqs_str, log_scores = decoder.beam_probs(
                            probs[i, :T].detach().cpu().numpy(), beam_size=5
                        )
                        best_idx = torch.argmax(log_scores).item()
                        pred_docs.append(seqs_str[best_idx])

            def chars_in_doc(doc: str):
                return {ch for ch in doc if ch not in (" ", "_")}

            N = len(labeled_docs) + len(pred_docs)
            df = Counter()
            for d in labeled_docs:
                df.update(chars_in_doc(d))
            for d in pred_docs:
                df.update(chars_in_doc(d))

            idf = {c: math.log((N + 1) / (df[c] + 1)) + 1.0 for c in df.keys()}

            scores_all = []
            for doc in pred_docs:
                if len(doc) == 0:
                    score = 0.0
                else:
                    cnt = Counter([ch for ch in doc if ch not in (" ", "_")])
                    L = sum(cnt.values())
                    s = 0.0
                    for ch, tf in cnt.items():
                        s += (tf / L) * idf.get(ch, math.log((N + 1) / 1) + 1.0)
                    score = s
                scores_all.append(torch.tensor([score], dtype=torch.float32))

            S = torch.cat(scores_all, dim=0).detach().cpu()
        
        elif self.method == "ideal":
            model.eval()
            scores_all = []
            
            df = pd.read_csv('best_score.csv')
            score_map = dict(zip(df["uid"].astype(str), df["letter_acc_beam"].astype(float)))

            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    _, _, _, _, data_info = batch
                    uids = data_info.get("uid")

                    scores_use = torch.tensor(
                        [score_map[str(u)] for u in uids],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    # x logx
                    x = scores_use.clamp_min(1e-12)
                    x = x/100.0
                    scores_use = -x * torch.log(x)

                    scores_all.append(scores_use)
            S=torch.cat(scores_all).detach().cpu()
        
        elif self.method == "ideal_tfidf":
            from collections import Counter
            import math

            # 1) 현재 labeled pool로부터 IDF 구축 (라벨 누출 방지)
            labeled_df = self._subset_to_dataframe(torch.utils.data.Subset(self.train_set, self.labeled_idxs))
            assert "phrase" in labeled_df.columns, "train df에 'phrase' 컬럼이 필요합니다."
            labeled_docs = [str(s) for s in labeled_df["phrase"].fillna("").tolist()]

            def chars_in_doc(doc: str):
                # DF(문서 빈도)용: 한 문서에 등장했는지 여부만 (공백/'_' 제외)
                return {ch for ch in doc if ch not in (" ", "_")}

            N = len(labeled_docs)
            df_c = Counter()
            for d in labeled_docs:
                df_c.update(chars_in_doc(d))

            # smooth IDF
            idf = {c: math.log((N + 1) / (df_c[c] + 1)) + 1.0 for c in df_c.keys()}
            default_idf = math.log((N + 1) / 1) + 1.0

            # 2) unlabeled는 GT로 TF를 계산(oracle) → 점수 산출
            scores_all = []
            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    # batch: (inputs, targets, input_lengths, target_lengths, data_info)
                    _, targets, _, target_lengths, _ = batch

                    if torch.is_tensor(targets):
                        B = targets.size(0)
                    else:
                        B = len(targets)

                    for b in range(B):
                        tgt_len = int(target_lengths[b].item())
                        tgt_idxs = targets[b][:tgt_len].detach().cpu().numpy().tolist()
                        gt = "".join([decoder.int_to_char[i] for i in tgt_idxs])

                        chars = [ch for ch in gt if ch not in (" ", "_")]
                        if not chars:
                            score = 0.0
                        else:
                            cnt = Counter(chars)           # TF(문자 카운트)
                            Lc  = sum(cnt.values())        # 문서 길이(문자수)
                            # TF(normalized) * IDF 합
                            score = sum((tf / Lc) * idf.get(ch, default_idf) for ch, tf in cnt.items())

                        scores_all.append(score)

            S = torch.tensor(scores_all, dtype=torch.float32).detach().cpu()
        
        elif self.method == "temporal_perturbation_acc2": # letter acc
            model.eval()
            scores_all = []

            # 윈도우 빌더: full, half-front, half-back, third-front, third-middle, third-back
            def build_windows(L: torch.Tensor, T: int, device: torch.device):
                L = L.clamp_min(1).clamp_max(T)         # (B,)
                h = L // 2
                t = L // 3
                zero = torch.zeros_like(L)
                s = torch.stack([zero, zero, h, zero, t, L - t], dim=1).to(device)   # (B,6)
                e = torch.stack([L,    h,    L,  t,  2*t, L],      dim=1).to(device) # (B,6)
                # 정규화(순서 중요)
                Lc = L.unsqueeze(1)
                s = torch.minimum(s, Lc - 1)   # s ≤ L-1
                e = torch.maximum(e, s + 1)    # 길이 ≥ 1
                e = torch.minimum(e, Lc)       # e ≤ L
                return s.to(torch.long), e.to(torch.long)

            def make_window_mask(s, e, T, device):
                t_idx = torch.arange(T, device=device).view(1, 1, T)  # (1,1,T)
                return ((t_idx >= s.unsqueeze(-1)) & (t_idx < e.unsqueeze(-1))).float()  # (B,6,T)

            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = inputs.to(self.device), input_lengths.to(self.device)
                    B, T = inputs.size(0), inputs.size(1)
                    L = input_lengths.clamp_max(T).to(torch.long)  # (B,)

                    # 1) s/e & 마스크
                    s, e = build_windows(L, T, device=self.device)          # (B,6)
                    mask = make_window_mask(s, e, T, device=self.device)    # (B,6,T)
                    W = s.size(1)  # 6

                    # 2) 배치 확장 + 0마스킹
                    x_rep  = inputs.unsqueeze(1).repeat(1, W, 1, 1, 1).view(B*W, T, *inputs.shape[2:])  # (B*W,T,54,3)
                    m_rep  = mask.unsqueeze(-1).unsqueeze(-1).view(B*W, T, 1, 1)                        # (B*W,T,1,1)
                    x_mask = x_rep * m_rep

                    # 3) left-align (각 윈도우 시작 s만큼 앞으로 당김 → [segment, 0, 0, ...])
                    s_flat = s.view(-1)                                        # (B*W,)
                    base   = torch.arange(T, device=self.device).view(1, T)    # (1,T)
                    gather_idx  = (base + s_flat.view(-1, 1)) % T              # (B*W,T)
                    gather_idx4 = gather_idx.view(B*W, T, 1, 1).expand(B*W, T, x_mask.size(2), x_mask.size(3))
                    x_shift = torch.gather(x_mask, dim=1, index=gather_idx4)   # (B*W,T,54,3)

                    # 4) 한 번에 forward
                    logits = model(x_shift)                     # (B*W,T,V)
                    probs  = F.softmax(logits, dim=-1)

                    # 5) full & 부분 윈도우를 greedy decode 하고, letter acc 평균 계산
                    lens_flat = (e - s).view(-1)               # (B*W,)
                    batch_scores = []                          # 각 샘플 b의 mean-acc(%) 저장

                    for b in range(B):
                        # full 윈도우(인덱스 0)의 디코드 → base_pred
                        base_part_preds = []
                        base_idx = b * W + 0
                        for w in range(1, W):  # 1..5 (부분 윈도우만)
                            si = int(s[b, w].item())
                            ei = int(e[b, w].item())                        
                            base_seq_np = probs[base_idx, si:ei].detach().cpu().numpy()
                            base_part_preds.append("".join(decoder.greedy_decode(base_seq_np)))

                        # 부분 윈도우(1..5) 디코드
                        part_preds = []
                        for w in range(1, W):
                            idx = b * W + w
                            li  = int(lens_flat[idx].item())
                            seg_np = probs[idx, :li].detach().cpu().numpy()
                            pred = "".join(decoder.greedy_decode(seg_np))
                            part_preds.append(pred)

                        # letter accuracy(%): base_pred와 부분 윈도우 결과들의 평균 정밀도
                        # compute_acc가 리스트 간 평균 % 리턴한다고 가정
                        if len(part_preds) == 0:
                            mean_acc = 100.0
                        else:
                            mean_acc = float(compute_acc(base_part_preds, part_preds))  # %
                        batch_scores.append(-mean_acc)

                    scores_all.append(torch.tensor(batch_scores, dtype=torch.float32, device=self.device))

            S = torch.cat(scores_all, dim=0).detach().cpu()

        
        elif self.method == "temporal_perturbation_std": # std
            model.eval()
            scores_all = []

            BEAM = 5

            def lc_conf_from_probs(prob_seg_np):
                seqs_str, log_scores = decoder.beam_probs(prob_seg_np, beam_size=BEAM)
                lens = torch.tensor([max(1, len(s)) for s in seqs_str], dtype=torch.float32)
                idx = torch.argmax(log_scores)
                best = torch.exp(log_scores[idx] / lens[idx].clamp_min(1.0))
                return (1.0 - best).to(torch.float32)

            def build_windows(L: torch.Tensor, T: int, device: torch.device):
                L = L.clamp_min(1).clamp_max(T)         # (B,)
                h = L // 2
                t = L // 3
                zero = torch.zeros_like(L)
                s = torch.stack([zero, zero, h, zero, t, L - t], dim=1).to(device)   # (B,6)
                e = torch.stack([L,    h,    L,  t,  2*t, L],      dim=1).to(device) # (B,6)
                # 정규화(순서 중요)
                Lc = L.unsqueeze(1)
                s = torch.minimum(s, Lc - 1)   # s ≤ L-1
                e = torch.maximum(e, s + 1)    # 길이 ≥ 1
                e = torch.minimum(e, Lc)       # e ≤ L
                return s.to(torch.long), e.to(torch.long)

            def make_window_mask(s, e, T, device):
                t_idx = torch.arange(T, device=device).view(1, 1, T)
                return ((t_idx >= s.unsqueeze(-1)) & (t_idx < e.unsqueeze(-1))).float()  # (B, W, T)

            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = inputs.to(self.device), input_lengths.to(self.device)
                    B, T = inputs.size(0), inputs.size(1)
                    L = input_lengths.clamp_max(T)

                    # 1) 윈도우 정의 + 마스크
                    s, e = build_windows(L, T, device=self.device)              # (B, 6)
                    mask = make_window_mask(s, e, T, device=self.device)        # (B, 6, T)

                    # 2) 배치 확장 후 0마스킹
                    x_rep  = inputs.unsqueeze(1).repeat(1, 6, 1, 1, 1).view(B*6, T, *inputs.shape[2:])  # (B*6, T, 54, 3)
                    m_rep  = mask.unsqueeze(-1).unsqueeze(-1).view(B*6, T, 1, 1)                        # (B*6, T, 1, 1)
                    x_mask = x_rep * m_rep

                    # 3) **앞으로 당기기(Left-align)**: 시작 s만큼 왼쪽으로 롤 → [segment, 0, 0, ...]
                    s_flat = s.view(-1)                                        # (B*6,)
                    base   = torch.arange(T, device=self.device).view(1, T)    # (1, T)
                    # roll left by s: gather index = (t + s) % T
                    gather_idx = (base + s_flat.view(-1,1)) % T                # (B*6, T)
                    gather_idx4 = gather_idx.view(B*6, T, 1, 1).expand(B*6, T, x_mask.size(2), x_mask.size(3))
                    x_shift = torch.gather(x_mask, dim=1, index=gather_idx4)   # (B*6, T, 54, 3)

                    # 4) 한 번에 forward
                    logits = model(x_shift)                                     # (B*6, T, V)
                    probs  = F.softmax(logits, dim=-1)

                    # 5) 각 윈도우 길이만큼 **앞쪽**을 잘라 confidence 계산
                    lens_flat = (e - s).view(-1)                                # (B*6,)
                    conf_list = []
                    for i in range(B*6):
                        li = int(lens_flat[i].item())
                        seg_np = probs[i, :li].detach().cpu().numpy()
                        conf_list.append(lc_conf_from_probs(seg_np))
                    confs = torch.stack(conf_list, dim=0).to(self.device).view(B, 6)

                    score = confs.std(dim=1, unbiased=False)
                    scores_all.append(score)

            S = torch.cat(scores_all, dim=0).detach().cpu()

        elif self.method == "tsc_ht_noshift_std":
            model.eval()
            scores_all = []

            BEAM = 5

            def lc_conf_from_probs(prob_seg_np):
                seqs_str, log_scores = decoder.beam_probs(prob_seg_np, beam_size=BEAM)
                lens = torch.tensor([max(1, len(s)) for s in seqs_str], dtype=torch.float32)
                idx = torch.argmax(log_scores)
                best = torch.exp(log_scores[idx] / lens[idx].clamp_min(1.0))
                return (1.0 - best).to(torch.float32)

            def build_windows(L: torch.Tensor, T: int, device: torch.device):
                L = L.clamp_min(1).clamp_max(T)         # (B,)
                h = L // 2
                t = L // 3
                zero = torch.zeros_like(L)
                s = torch.stack([zero, zero, h, zero, t, L - t], dim=1).to(device)   # (B,6)
                e = torch.stack([L,    h,    L,  t,  2*t, L],      dim=1).to(device) # (B,6)
                # 정규화(순서 중요)
                Lc = L.unsqueeze(1)
                s = torch.minimum(s, Lc - 1)
                e = torch.maximum(e, s + 1)
                e = torch.minimum(e, Lc)
                return s.to(torch.long), e.to(torch.long)

            def make_window_mask(s, e, T, device):
                t_idx = torch.arange(T, device=device).view(1, 1, T)
                return ((t_idx >= s.unsqueeze(-1)) & (t_idx < e.unsqueeze(-1))).float()  # (B, W, T)

            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = inputs.to(self.device), input_lengths.to(self.device)
                    B, T = inputs.size(0), inputs.size(1)
                    L = input_lengths.clamp_max(T)

                    # 1) 윈도우 정의 + 마스크
                    s, e = build_windows(L, T, device=self.device)              # (B, 6)
                    mask = make_window_mask(s, e, T, device=self.device)        # (B, 6, T)

                    # 2) 배치 확장 후 0마스킹
                    x_rep  = inputs.unsqueeze(1).repeat(1, 6, 1, 1, 1).view(B*6, T, *inputs.shape[2:])  # (B*6, T, 54, 3)
                    m_rep  = mask.unsqueeze(-1).unsqueeze(-1).view(B*6, T, 1, 1)                        # (B*6, T, 1, 1)
                    x_mask = x_rep * m_rep

                    logits = model(x_mask)                                     # (B*6, T, V)
                    probs  = F.softmax(logits, dim=-1)

                    s_flat, e_flat = s.view(-1), e.view(-1)
                    conf_list = []
                    for i in range(B*6):
                        si, ei = int(s_flat[i].item()), int(e_flat[i].item())
                        seg_np = probs[i, si:ei].detach().cpu().numpy()
                        conf_list.append(lc_conf_from_probs(seg_np))
                    confs = torch.stack(conf_list, dim=0).to(self.device).view(B, 6)

                    score = confs.std(dim=1, unbiased=False)
                    scores_all.append(score)

            S = torch.cat(scores_all, dim=0).detach().cpu()
        
        elif self.method == "tsc_htback_leftshift_std":
            model.eval()
            scores_all = []

            BEAM = 5

            def lc_conf_from_probs(prob_seg_np):
                seqs_str, log_scores = decoder.beam_probs(prob_seg_np, beam_size=BEAM)
                lens = torch.tensor([max(1, len(s)) for s in seqs_str], dtype=torch.float32)
                idx = torch.argmax(log_scores)
                best = torch.exp(log_scores[idx] / lens[idx].clamp_min(1.0))
                return (1.0 - best).to(torch.float32)

            def build_windows(L: torch.Tensor, T: int, device: torch.device):
                L = L
                h = L // 2
                t = L // 3
                zero = torch.zeros_like(L)
                s = torch.stack([zero, h, t, L - t], dim=1).to(device)
                e = torch.stack([L,    L, L, L    ], dim=1).to(device)
                # 정규화(순서 중요)
                Lc = L.unsqueeze(1)
                s = torch.minimum(s, Lc - 1)   # s ≤ L-1
                e = torch.maximum(e, s + 1)    # 길이 ≥ 1
                e = torch.minimum(e, Lc)       # e ≤ L
                return s.to(torch.long), e.to(torch.long)

            def make_window_mask(s, e, T, device):
                t_idx = torch.arange(T, device=device).view(1, 1, T)
                return ((t_idx >= s.unsqueeze(-1)) & (t_idx < e.unsqueeze(-1))).float()  # (B, W, T)

            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = inputs.to(self.device), input_lengths.to(self.device)
                    B, T = inputs.size(0), inputs.size(1)
                    L = input_lengths

                    # 1) 윈도우 정의 + 마스크
                    s, e = build_windows(L, T, device=self.device)
                    mask = make_window_mask(s, e, T, device=self.device)
                    W = s.size(1)

                    # 2) 배치 확장 후 0마스킹
                    x_rep  = inputs.unsqueeze(1).repeat(1, W, 1, 1, 1).view(B*W, T, *inputs.shape[2:])
                    m_rep  = mask.unsqueeze(-1).unsqueeze(-1).view(B*W, T, 1, 1)
                    x_mask = x_rep * m_rep

                    # 3) left-shift
                    s_flat = s.view(-1)
                    base   = torch.arange(T, device=self.device).view(1, T)
                    
                    gather_idx = (base + s_flat.view(-1,1)) % T
                    gather_idx4 = gather_idx.view(B*W, T, 1, 1).expand(B*W, T, x_mask.size(2), x_mask.size(3))
                    x_shift = torch.gather(x_mask, dim=1, index=gather_idx4)

                    # 4) 한 번에 forward
                    logits = model(x_shift)
                    probs  = F.softmax(logits, dim=-1)

                    # 5) 각 윈도우 길이만큼 앞쪽을 잘라 confidence 계산
                    lens_flat = (e - s).view(-1)
                    conf_list = []
                    for i in range(B*W):
                        li = int(lens_flat[i].item())
                        seg_np = probs[i, :li].detach().cpu().numpy()
                        conf_list.append(lc_conf_from_probs(seg_np))
                    confs = torch.stack(conf_list, dim=0).to(self.device).view(B, W)

                    score = confs.std(dim=1, unbiased=False)
                    scores_all.append(score)

            S = torch.cat(scores_all, dim=0).detach().cpu()
        
        elif self.method == "tsc_frames_leftshift_std":
            model.eval()
            scores_all = []

            BEAM = 5
            CHUNK = 16

            def lc_conf_from_probs(prob_seg_np):
                # Least Confidence = 1 - exp(top_logprob / length)
                seqs_str, log_scores = decoder.beam_probs(prob_seg_np, beam_size=BEAM)
                lens = torch.tensor([max(1, len(s)) for s in seqs_str], dtype=torch.float32)
                idx = torch.argmax(log_scores)
                best = torch.exp(log_scores[idx] / lens[idx].clamp_min(1.0))
                return (1.0 - best).to(torch.float32)

            def make_window_mask(s, e, T, device):
                # s,e: (B,Wc) → mask: (B,Wc,T) with 1 on [s,e)
                t_idx = torch.arange(T, device=device).view(1, 1, T)
                return ((t_idx >= s.unsqueeze(-1)) & (t_idx < e.unsqueeze(-1))).float()

            with torch.no_grad():
                for batch in tqdm(unlabeled_loader):
                    inputs, _, input_lengths, _, _ = batch
                    inputs, input_lengths = inputs.to(self.device), input_lengths.to(self.device)
                    B, T = inputs.size(0), inputs.size(1)
                    L = input_lengths.to(torch.long)

                    # 배치 단위로 std를 누적 집계 (가변 개수 윈도우를 청크로 처리)
                    cnt   = torch.zeros(B, dtype=torch.long, device=self.device)
                    ssum  = torch.zeros(B, dtype=torch.float32, device=self.device)
                    ssum2 = torch.zeros(B, dtype=torch.float32, device=self.device)

                    # 시작 프레임 s = 0..T-1 을 CHUNK 단위로 순회 (유효성은 샘플별 L로 필터)
                    for c0 in range(0, T, CHUNK):
                        c1 = min(T, c0 + CHUNK)
                        Wc = c1 - c0  # 이 청크의 윈도우 개수

                        # s_chunk: (B, Wc) = [c0..c1-1] broadcast, e_chunk: (B, Wc) = L[b]
                        s_chunk = torch.arange(c0, c1, device=self.device).view(1, Wc).expand(B, -1)
                        e_chunk = L.unsqueeze(1).expand_as(s_chunk)

                        # 유효 윈도우(샘플 b에서 s < L[b])만 사용
                        valid_chunk = (s_chunk < L.unsqueeze(1))  # (B, Wc) bool

                        # 최소 길이 ≥1 보장 (무효 칸은 어차피 버리지만 index 안정화 차원에서 처리)
                        s_chunk = torch.minimum(s_chunk, L.unsqueeze(1) - 1)
                        e_chunk = torch.maximum(e_chunk, s_chunk + 1)
                        e_chunk = torch.minimum(e_chunk, L.unsqueeze(1))

                        # 마스크 만들고 입력 확장/적용
                        mask = make_window_mask(s_chunk, e_chunk, T, device=self.device)  # (B,Wc,T)
                        x_rep = inputs.unsqueeze(1).repeat(1, Wc, 1, 1, 1).view(B * Wc, T, *inputs.shape[2:])
                        m_rep = mask.unsqueeze(-1).unsqueeze(-1).view(B * Wc, T, 1, 1)
                        x_mask = x_rep * m_rep

                        # invalid 윈도우는 통째로 0으로 죽여서 모델에 영향 최소화
                        inv = (~valid_chunk).view(B * Wc, 1, 1, 1).to(x_mask.dtype)
                        x_mask = x_mask * (1.0 - inv)

                        # left-shift: 각 윈도우 시작 s만큼 앞으로 당겨 [segment, 0, 0, ...]
                        s_flat = s_chunk.view(-1)  # (B*Wc,)
                        base   = torch.arange(T, device=self.device).view(1, T)
                        gather_idx  = (base + s_flat.view(-1, 1)) % T
                        gather_idx4 = gather_idx.view(B * Wc, T, 1, 1).expand(B * Wc, T, x_mask.size(2), x_mask.size(3))
                        x_shift = torch.gather(x_mask, dim=1, index=gather_idx4)  # (B*Wc, T, 54, 3)

                        # forward
                        logits = model(x_shift)                  # (B*Wc, T, V)
                        probs  = F.softmax(logits, dim=-1)

                        # 각 윈도우 길이만큼 앞쪽을 잘라 confidence 계산
                        lens_flat = (e_chunk - s_chunk).view(-1)  # (B*Wc,)
                        confs = []
                        for i in range(B * Wc):
                            if not valid_chunk.view(-1)[i]:
                                confs.append(torch.tensor(0.0, device=self.device))  # 자리 채움(집계 시 무시)
                                continue
                            li = int(lens_flat[i].item())
                            seg_np = probs[i, :li].detach().cpu().numpy()
                            confs.append(lc_conf_from_probs(seg_np).to(self.device))
                        confs = torch.stack(confs, dim=0)  # (B*Wc,)

                        # per-sample로 count/sum/sum2 누적 (scatter_add)
                        sample_idx = torch.arange(B, device=self.device).repeat_interleave(Wc)  # (B*Wc,)
                        valid_flat = valid_chunk.view(-1)

                        # count
                        tmpc = torch.zeros(B, dtype=torch.long, device=self.device)
                        cnt  = cnt + tmpc.scatter_add(0, sample_idx, valid_flat.to(torch.long))
                        # sum
                        tmps = torch.zeros(B, dtype=torch.float32, device=self.device)
                        ssum = ssum + tmps.scatter_add(0, sample_idx, confs * valid_flat.to(confs.dtype))
                        # sum of squares
                        tmps2 = torch.zeros(B, dtype=torch.float32, device=self.device)
                        ssum2 = ssum2 + tmps2.scatter_add(0, sample_idx, (confs * valid_flat.to(confs.dtype))**2)

                    # 배치 내 각 샘플 std 계산 (윈도우가 1개면 std=0)
                    denom = cnt.clamp_min(1).to(torch.float32)
                    mean  = ssum / denom
                    var   = ssum2 / denom - mean**2
                    var   = torch.clamp(var, min=0.0)
                    std   = torch.sqrt(var)  # (B,)
                    scores_all.append(std)

            # 모든 배치 스코어 연결 후 Top-K (std 큰 순)
            S = torch.cat(scores_all, dim=0).detach().cpu()
        
        top_scores, topk_idxs = torch.topk(S, self.budget)
        self.selected = self.unlabeled_idxs[topk_idxs.detach().cpu().numpy()]
        scores = pd.Series(top_scores.numpy(), index=self.selected, dtype="float32")


        self.labeled_idxs = np.concatenate([self.labeled_idxs, self.selected])
        mask = ~np.isin(self.unlabeled_idxs, self.selected)
        self.unlabeled_idxs = self.unlabeled_idxs[mask]

        self.num_labeled += self.budget

        labeled_pool   = torch.utils.data.Subset(self.train_set, self.labeled_idxs)
        unlabeled_pool = torch.utils.data.Subset(self.train_set, self.unlabeled_idxs)

        self.stage += 1

        self._save_labeled_df(labeled_pool, scores, self.stage)

        return labeled_pool, unlabeled_pool