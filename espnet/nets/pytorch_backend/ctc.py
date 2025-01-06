import numpy as np
import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import to_device


class CTC(torch.nn.Module):
    """CTC module

    :param int odim: dimension of outputs
    :param int eprojs: number of encoder projection units
    :param float dropout_rate: dropout rate (0.0 ~ 1.0)
    :param bool reduce: reduce the CTC loss into a scalar
    """

    def __init__(self, odim, eprojs, dropout_rate, reduce=True):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.loss = None
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.probs = None  # for visualization

        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(
            reduction=reduction_type, zero_infinity=True
        )
        self.ignore_id = -1
        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen):
        th_pred = th_pred.log_softmax(2)
        with torch.backends.cudnn.flags(deterministic=True):
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        # Batch-size average
        loss = loss / th_pred.size(1)
        return loss

    def forward(self, hs_pad, hlens, ys_pad):
        """CTC forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad:
            batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys

        # zero padding for hs
        ys_hat = self.ctc_lo(self.dropout(hs_pad))
        ys_hat = ys_hat.transpose(0, 1)

        olens = to_device(ys_hat, torch.LongTensor([len(s) for s in ys]))
        hlens = hlens.long()
        ys_pad = torch.cat(ys)  # without this the code breaks for asr_mix
        self.loss = self.loss_fn(ys_hat, ys_pad, hlens, olens)

        if self.reduce:
            self.loss = self.loss.sum()

        return self.loss, ys_hat

    def softmax(self, hs_pad):
        """softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        self.probs = F.softmax(self.ctc_lo(hs_pad), dim=-1)
        return self.probs

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: log softmax applied 3d tensor (B, Tmax, odim)
        :rtype: torch.Tensor
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=-1)

    def argmax(self, hs_pad):
        """argmax of frame activations

        :param torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: torch.Tensor
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=-1)

    def forced_align(self, h, y, blank_id=0):
        """forced alignment.

        :param torch.Tensor h: hidden state sequence, 2d tensor (T, D)
        :param torch.Tensor y: id sequence tensor 1d tensor (L)
        :param int y: blank symbol index
        :return: best alignment results
        :rtype: list
        """

        def interpolate_blank(label, blank_id=0):
            """Insert blank token between every two label token."""
            label = np.expand_dims(label, 1)
            blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
            label = np.concatenate([blanks, label], axis=1)
            label = label.reshape(-1)
            label = np.append(label, label[0])
            return label

        lpz = self.log_softmax(h)
        lpz = lpz.squeeze(0)

        y_int = interpolate_blank(y, blank_id)

        logdelta = np.zeros((lpz.size(0), len(y_int))) - 100000000000.0  # log of zero
        state_path = (
            np.zeros((lpz.size(0), len(y_int)), dtype=np.int16) - 1
        )  # state path

        logdelta[0, 0] = lpz[0][y_int[0]]
        logdelta[0, 1] = lpz[0][y_int[1]]

        for t in range(1, lpz.size(0)):
            for s in range(len(y_int)):
                if y_int[s] == blank_id or s < 2 or y_int[s] == y_int[s - 2]:
                    candidates = np.array([logdelta[t - 1, s], logdelta[t - 1, s - 1]])
                    prev_state = [s, s - 1]
                else:
                    candidates = np.array(
                        [
                            logdelta[t - 1, s],
                            logdelta[t - 1, s - 1],
                            logdelta[t - 1, s - 2],
                        ]
                    )
                    prev_state = [s, s - 1, s - 2]
                logdelta[t, s] = np.max(candidates) + lpz[t][y_int[s]]
                state_path[t, s] = prev_state[np.argmax(candidates)]

        state_seq = -1 * np.ones((lpz.size(0), 1), dtype=np.int16)

        candidates = np.array(
            [logdelta[-1, len(y_int) - 1], logdelta[-1, len(y_int) - 2]]
        )
        prev_state = [len(y_int) - 1, len(y_int) - 2]
        state_seq[-1] = prev_state[np.argmax(candidates)]
        for t in range(lpz.size(0) - 2, -1, -1):
            state_seq[t] = state_path[t + 1, state_seq[t + 1, 0]]

        output_state_seq = []
        for t in range(0, lpz.size(0)):
            output_state_seq.append(y_int[state_seq[t, 0]])

        return output_state_seq

    def forced_align_batch(self, hs_pad, ys_pad, ilens, blank_id=0):
        """forced alignment with batch processing.

        :param torch.Tensor hs_pad: hidden state sequence, 3d tensor (T, B, D)
        :param torch.Tensor ys_pad: id sequence tensor 2d tensor (B, L)
        :param torch.Tensor ilens: Input length of each utterance (B,)
        :param int blank_id: blank symbol index
        :return: best alignment results
        :rtype: list of numpy.array
        """

        def interpolate_blank(label, olens_int):
            """Insert blank token between every two label token."""
            lab_len = label.shape[1] * 2 + 1
            label_out = np.full((label.shape[0], lab_len), blank_id, dtype=np.int64)
            label_out[:, 1::2] = label
            for b in range(label.shape[0]):
                label_out[b, olens_int[b] * 2 + 1 :] = self.ignore_id
            return label_out

        neginf = float("-inf")  # log of zero
        # lpz = self.log_softmax(hs_pad).cpu().detach().numpy()
        # hs_pad = hs_pad.transpose(1,0)
        lpz = F.log_softmax(hs_pad, dim=-1).cpu().detach().numpy()
        ilens = ilens.cpu().detach().numpy()

        ys_pad = ys_pad.cpu().detach().numpy()
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        olens = np.array([len(s) for s in ys])
        olens_int = olens * 2 + 1
        ys_int = interpolate_blank(ys_pad, olens_int)

        Tmax, B, _ = lpz.shape
        Lmax = ys_int.shape[-1]
        logdelta = np.full((Tmax, B, Lmax), neginf, dtype=lpz.dtype)
        state_path = -np.ones(logdelta.shape, dtype=np.int16)  # state path

        b_indx = np.arange(B, dtype=np.int64)
        t_0 = np.zeros(B, dtype=np.int64)
        logdelta[0, :, 0] = lpz[t_0, b_indx, ys_int[:, 0]]
        logdelta[0, :, 1] = lpz[t_0, b_indx, ys_int[:, 1]]

        s_indx_mat = np.arange(Lmax)[None, :].repeat(B, 0)
        notignore_mat = ys_int != self.ignore_id
        same_lab_mat = np.zeros((B, Lmax), dtype=bool)
        same_lab_mat[:, 3::2] = ys_int[:, 3::2] == ys_int[:, 1:-2:2]
        Lmin = olens_int.min()
        for t in range(1, Tmax):
            s_start = max(0, Lmin - (Tmax - t) * 2)
            s_end = min(Lmax, t * 2 + 2)
            candidates = np.full((B, Lmax, 3), neginf, dtype=logdelta.dtype)
            candidates[:, :, 0] = logdelta[t - 1, :, :]
            candidates[:, 1:, 1] = logdelta[t - 1, :, :-1]
            candidates[:, 3::2, 2] = logdelta[t - 1, :, 1:-2:2]
            candidates[same_lab_mat, 2] = neginf
            candidates_ = candidates[:, s_start:s_end, :]
            idx = candidates_.argmax(-1)
            b_i, s_i = np.ogrid[:B, : idx.shape[-1]]
            nignore = notignore_mat[:, s_start:s_end]
            logdelta[t, :, s_start:s_end][nignore] = (
                candidates_[b_i, s_i, idx][nignore]
                + lpz[t, b_i, ys_int[:, s_start:s_end]][nignore]
            )
            s = s_indx_mat[:, s_start:s_end]
            state_path[t, :, s_start:s_end][nignore] = (s - idx)[nignore]

        alignments = []
        prev_states = logdelta[
            ilens[:, None] - 1,
            b_indx[:, None],
            np.stack([olens_int - 2, olens_int - 1], -1),
        ].argmax(-1)
        for b in range(B):
            T, L = ilens[b], olens_int[b]
            prev_state = prev_states[b] + L - 2
            ali = np.empty(T, dtype=ys_int.dtype)
            ali[T - 1] = ys_int[b, prev_state]
            for t in range(T - 2, -1, -1):
                prev_state = state_path[t + 1, b, prev_state]
                ali[t] = ys_int[b, prev_state]
            alignments.append(ali)

        return alignments
