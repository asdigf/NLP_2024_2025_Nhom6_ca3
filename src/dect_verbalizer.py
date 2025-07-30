from inspect import Parameter
import json
import time
from os import stat
import os
from transformers.file_utils import ModelOutput
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.data_utils import InputExample, InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, MaskedLMOutput

class DecTVerbalizer(Verbalizer):
    r"""
    Lớp thực hiện Verbalizer trong `Prototypical Verbalizer for Prompt-based Few-shot Tuning`

    Args:   
        tokenizer (:obj:`PreTrainedTokenizer`): Bộ mã hóa của mô hình pre-trained hiện tại để chỉ định từ vựng.
        classes (:obj:`List[Any]`): Các lớp (hoặc nhãn) của nhiệm vụ hiện tại.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): Các từ nhãn được ánh xạ bởi các nhãn.
        prefix (:obj:`str`, optional): Chuỗi tiền tố của verbalizer (dùng trong PLM như RoBERTa, nhạy với khoảng trắng tiền tố).
        multi_token_handler (:obj:`str`, optional): Chiến lược xử lý cho nhiều token được tạo bởi tokenizer.
        post_log_softmax (:obj:`bool`, optional): Có áp dụng log softmax sau xử lý trên label_logits không. Mặc định là True.
        lr: (:obj:`float`, optional): Tốc độ học cho prototype.
        hidden_size: (:obj:`int`, optional): Kích thước của trạng thái ẩn của mô hình.
        mid_dim: (:obj:`int`, optional): Kích thước của embedding prototype.
        epochs: (:obj:`int`, optional): Số epoch huấn luyện prototype.
        model_logits_weight: (:obj:`float`, optional): Hệ số trọng số (\lambda) cho model logits.
    """
    def __init__(self, 
                tokenizer: Optional[PreTrainedTokenizer],
                classes: Optional[List] = None,
                num_classes: Optional[int] = None,
                label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                prefix: Optional[str] = "",
                multi_token_handler: Optional[str] = "first",
                post_log_softmax: Optional[bool] = True,
                lr: Optional[float] = 1e-3,
                hidden_size: Optional[int] = 1024,
                mid_dim: Optional[int] = 64,
                epochs: Optional[int] = 5,
                model_logits_weight: Optional[float] = 1,
                save_dir: Optional[str] = None,
                model: Optional[nn.Module] = None
                ):
        
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.post_log_softmax = post_log_softmax
        self.lr = lr
        self.mid_dim = mid_dim
        self.epochs = epochs
        self.model_logits_weight = model_logits_weight
        self.save_dir = save_dir
        self.hidden_dims = hidden_size
        self.num_label_words_per_class = None

        # Khởi tạo projection head
        self.head = nn.Linear(self.hidden_dims, self.mid_dim, bias=False)

        # Nếu mô hình dùng float16 → ép dtype cho self.head
        if model is not None and next(model.parameters()).dtype == torch.float16:
            self.head = self.head.half()
            self.proto = self.proto.half()
            self.proto_r = self.proto_r.half()

        # Nếu có label words để khởi tạo prototype
        if label_words is not None:
            self.label_words = label_words

        # Proto vector
        w = torch.empty((self.num_classes, self.mid_dim))
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=False)

        # Bán kính của prototype
        r = torch.ones(self.num_classes)
        self.proto_r = nn.Parameter(r, requires_grad=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    @property
    def group_parameters_proto(self,):
        r"""Bao gồm các tham số của lớp cuối cùng
        """
        return [p for n, p in self.head.named_parameters()] + [self.proto_r]

    def on_label_words_set(self):
        self.label_words = self.add_prefix(self.label_words, self.prefix)
        self.generate_parameters()
        
    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Thêm tiền tố vào các từ nhãn. Ví dụ, nếu một từ nhãn nằm giữa template,
        tiền tố nên là ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): Các từ nhãn được ánh xạ bởi các nhãn.
            prefix (:obj:`str`, optional): Chuỗi tiền tố của verbalizer.
        
        Returns:
            :obj:`Sequence[str]`: Các từ nhãn mới kèm tiền tố.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  # Bao bọc thành danh sách các danh sách từ nhãn.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""Trong template thủ công cơ bản, các tham số được tạo trực tiếp từ các từ nhãn.
        Trong triển khai này, các từ nhãn không nên được mã hóa thành nhiều hơn một token.
        """
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        max_len = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        self.num_label_words_per_class = max_num_label_words
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label)) 
                             for ids_per_label in all_ids]
        
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # Mặt nạ 3 chiều
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)

    def process_hiddens(self, hiddens: torch.Tensor, model_logits, **kwargs):
        r"""Khung xử lý toàn bộ logits gốc trên từ vựng, gồm bốn bước:
        """
        proto_logits = self.sim(self.head(hiddens), self.proto, self.proto_r, model_logits, self.model_logits_weight)
        return proto_logits

    def project(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        print("🔍 Hình dạng logits:", logits.shape)

        # Lọc bỏ các token đệm (số 0) từ label_words_ids
        label_ids = []
        for ids_per_label in self.label_words_ids:
            for ids in ids_per_label:
                # Chỉ bao gồm các ID token khác 0
                valid_ids = [id for id in ids if id != 0]
                if valid_ids:
                    label_ids.extend(valid_ids)

        print("✅ Sử dụng các ID token nhãn:", label_ids)

        if logits.ndim < 2 or logits.shape[1] == 0:
            print("⚠️ Logits không hợp lệ: shape nhỏ hơn 2 chiều hoặc chiều thứ 2 bằng 0.")
            return torch.empty(logits.shape[0], self.num_classes, device=logits.device, dtype=logits.dtype)

        # Chọn logits cho các ID token nhãn hợp lệ
        label_words_logits = logits[:, label_ids]
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""Khung xử lý toàn bộ logits gốc trên từ vựng, gồm bốn bước:

        (1) Ánh xạ logits thành logits của các từ nhãn

        nếu self.post_log_softmax là True:

            (2) Chuẩn hóa trên tất cả các từ nhãn

            (3) Hiệu chỉnh (tùy chọn)

        (4) Tổng hợp (cho nhiều từ nhãn)

        Args:
            logits (:obj:`torch.Tensor`): Logits gốc.
        
        Returns:
            (:obj:`torch.Tensor`): Logits cuối cùng đã xử lý trên các nhãn (lớp).
        """
        # Ánh xạ
        if logits.ndim < 2 or logits.shape[1] == 0:
            print("⚠️ Logits không hợp lệ: shape nhỏ hơn 2 chiều hoặc chiều thứ 2 bằng 0.")
            return torch.empty(logits.shape[0], self.num_classes, device=logits.device, dtype=logits.dtype)

        label_words_logits = self.project(logits, **kwargs)

        if self.post_log_softmax:
            if hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                # Đảm bảo shape khớp
                if self._calibrate_logits.shape[1:] == label_words_logits.shape[1:]:
                    label_words_logits = self.calibrate(label_words_probs=label_words_logits)
                else:
                    print("⚠️ Bỏ qua hiệu chỉnh vì shape không khớp:",
                        f"{self._calibrate_logits.shape} vs {label_words_logits.shape}")

        label_logits = self.aggregate(label_words_logits)
        if label_logits.dim() == 1:
            label_logits = label_logits.unsqueeze(0)  # Chuyển [K] -> [1, K]
        return label_logits
    
    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Cho logits trên toàn bộ từ vựng, trả về xác suất trên tập các từ nhãn.
        
        Args:
            logits (:obj:`Tensor`): Logits trên toàn bộ từ vựng.

        Returns:
            :obj:`Tensor`: Logits trên tập các từ nhãn.
        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)

    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        # Tính số lượng ID token hợp lệ cho mỗi lớp
        valid_ids_per_class = [sum(1 for id in ids if id != 0) for ids in self.label_words_ids.view(self.num_classes, -1)]
        max_valid_ids = max(valid_ids_per_class)
        
        # Khởi tạo tensor đầu ra
        batch_size = label_words_logits.shape[0]
        aggregated = torch.zeros(batch_size, self.num_classes, device=label_words_logits.device, dtype=label_words_logits.dtype)
        
        # Chỉ số để theo dõi vị trí trong label_words_logits
        idx = 0
        for i in range(self.num_classes):
            num_valid_ids = valid_ids_per_class[i]
            if num_valid_ids > 0:
                # Trích xuất logits cho lớp này
                class_logits = label_words_logits[:, idx:idx + num_valid_ids]
                # Tổng hợp bằng cách lấy trung bình các logits hợp lệ
                aggregated[:, i] = class_logits.mean(dim=-1)
                idx += num_valid_ids

        return aggregated

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""
        
        Args:
            label_words_probs (:obj:`torch.Tensor`): Phân phối xác suất của các từ nhãn với hình dạng [``batch_size``, ``num_classes``, ``num_label_words_per_class``]
        
        Returns:
            :obj:`torch.Tensor`: Xác suất đã hiệu chỉnh của các từ nhãn.
        """
        shape = label_words_probs.shape
        calibrate_label_words_probs = self._calibrate_logits
        if calibrate_label_words_probs.dim() == 2 and label_words_probs.dim() == 3:
            calibrate_label_words_probs = calibrate_label_words_probs.unsqueeze(-1)

        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape không khớp"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        
        return label_words_probs

    def process_outputs(self, outputs: Union[torch.Tensor, torch.Tensor], batch: Union[Dict, InputFeatures], **kwargs):
        model_logits = self.process_logits(outputs[1])
        proto_logits = self.process_hiddens(outputs[0], model_logits)
        return proto_logits

    def gather_outputs(self, outputs: ModelOutput):
        logits = outputs.logits
        if isinstance(outputs, Seq2SeqLMOutput):
            ret = outputs.decoder_hidden_states[-1]
        elif isinstance(outputs, MaskedLMOutput) or isinstance(outputs, CausalLMOutputWithCrossAttentions):
            ret = outputs.hidden_states[-1]
        else:
            try:
                ret = outputs.hidden_states[-1]
            except AttributeError:
                raise NotImplementedError(f"Phương thức gather outputs cho loại outputs {type(outputs)} chưa được triển khai")

        return ret, logits

    @staticmethod
    def sim(x, y, r=0, model_logits=0, model_logits_weight=1):
        if x.dim() == 3:
            x = x.mean(dim=1)
        
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        
        sim_matrix = torch.mm(x, y.t())
        
        # Đảm bảo model_logits khớp với kích thước sim_matrix
        if isinstance(model_logits, torch.Tensor):
            if model_logits.dim() == 1:
                model_logits = model_logits.unsqueeze(-1)
            if model_logits.shape[0] != sim_matrix.shape[0]:
                model_logits = model_logits[:sim_matrix.shape[0]]
        
        # Xử lý chiều của r
        if isinstance(r, torch.Tensor) and r.dim() == 0:
            r = r.unsqueeze(0)
        
        # Tính kết quả
        result = sim_matrix - model_logits * model_logits_weight - r
        return -result

    def loss_func(self, x, model_logits, labels):
        assert model_logits.shape[1] == self.num_classes, \
            f"❌ Hình dạng model_logits {model_logits.shape} không khớp num_classes = {self.num_classes}"
        sim_mat = torch.exp(self.sim(x, self.proto, self.proto_r, model_logits, self.model_logits_weight))
        pos_score = torch.sum(sim_mat * F.one_hot(labels), -1)
        loss = -torch.mean(torch.log(pos_score / sim_mat.sum(-1)))
        
        return loss

    def test(self, model, dataloader):
        batch_size = dataloader.batch_size
        model.eval()
        
        if dataloader is None:
            print("❌ Lỗi: Test dataloader không tồn tại (None)")
            return [], [], []
        
        if len(dataloader) == 0:
            print("❌ Lỗi: Test dataloader rỗng (0 samples)")
            return [], [], []
        
        print(f"🔍 Bắt đầu inference với {len(dataloader)} batch...")
        model_preds, preds, labels = [], [], []
        
        if os.path.isfile(f"{self.save_dir}/logits.pt"):
            print("📂 Đang tải logits và hiddens từ cache...")
            logits = torch.load(f"{self.save_dir}/logits.pt")
            hiddens = torch.load(f"{self.save_dir}/hiddens.pt")
            
            for i, batch in enumerate(dataloader):
                print(f"  → Đang xử lý batch {i+1}/{len(dataloader)}")
                try:
                    batch = batch.cuda().to_dict()
                    length = len(batch['label'])
                    labels.extend(batch.pop('label').cpu().tolist())
                    batch_hidden = hiddens[i*batch_size: i*batch_size+length].to(self.head.weight.dtype)
                    batch_logits = logits[i*batch_size: i*batch_size+length].to(self.head.weight.dtype)
                    proto = self.process_hiddens(batch_hidden, batch_logits)
                    model_pred = torch.argmax(batch_logits, dim=-1)
                    pred = torch.argmax(proto, dim=-1)
                    preds.extend(pred.cpu().tolist())
                    model_preds.extend(model_pred.cpu().tolist())
                except Exception as e:
                    print(f"❌ Lỗi khi xử lý batch {i+1}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        else:
            logits, hiddens = [], []
            print("🧠 Đang tính toán logits và hiddens mới...")
            with torch.no_grad(), torch.cuda.amp.autocast():
                total_batches = len(dataloader)
                for i, batch in enumerate(dataloader):
                    print(f"  → Đang xử lý batch {i+1}/{total_batches}")
                    try:
                        batch = batch.cuda().to_dict()
                        labels.extend(batch.pop('label').cpu().tolist())
                        outputs = model.prompt_model(batch)
                        outputs = self.gather_outputs(outputs)
                        batch_hidden = outputs[0][:, -1, :].to(self.head.weight.dtype)
                        batch_logits = outputs[1][:, -1, :].to(self.head.weight.dtype)
                        if batch_logits.shape[1] == 0:
                            print(f"⚠️ Bỏ qua batch {i+1} vì không có token để trích xuất.")
                            continue
                        model_logits = self.process_logits(batch_logits)
                        logits.append(model_logits)
                        hiddens.append(batch_hidden)
                        proto = self.process_hiddens(batch_hidden, model_logits)
                        model_pred = torch.argmax(model_logits, dim=-1)
                        pred = torch.argmax(proto, dim=-1)
                        preds.extend(pred.cpu().tolist())
                        model_preds.extend(model_pred.cpu().tolist())
                    except Exception as e:
                        print(f"❌ Lỗi khi xử lý batch {i+1}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if logits:
                    logits = torch.cat(logits, dim=0)
                    hiddens = torch.cat(hiddens, dim=0)
                    
                    os.makedirs(self.save_dir, exist_ok=True)
                    
                    print(f"💾 Đang lưu logits và hiddens vào {self.save_dir}")
                    torch.save(logits, f"{self.save_dir}/logits.pt")
                    torch.save(hiddens, f"{self.save_dir}/hiddens.pt")
                else:
                    print("⚠️ Không có logits nào được tạo do lỗi")
        
        print(f"✅ Đã hoàn thành inference: {len(labels)} mẫu")
        return model_preds, preds, labels

    def train_proto(self, model, dataloader, calibrate_dataloader):
        print("🧠 Verbalizer.label_words_ids:", self.label_words_ids)
        
        if hasattr(model, "tokenizer"):
            print("📚 Kích thước từ vựng từ tokenizer:", model.tokenizer.vocab_size)
        else:
            print("⚠️ Không tìm thấy tokenizer trong mô hình")

        model.eval()

        embeds = [[] for _ in range(self.num_classes)]
        labels = [[] for _ in range(self.num_classes)]
        model_logits_all = []

        total_num = 0
        start_time = time.time()

        all_embeds = []
        all_labels = []
        all_model_logits = []

        with torch.no_grad():
            # Hiệu chỉnh logits
            if calibrate_dataloader is not None:
                for i, batch in enumerate(calibrate_dataloader):
                    batch = batch.cuda().to_dict()
                    outputs = model.prompt_model(batch)
                    outputs = self.gather_outputs(outputs)
                    logits = self.project(outputs[1][:, -1, :].to(self.head.weight.dtype))
                    self._calibrate_logits = logits / torch.mean(logits)

            # Thu thập dữ liệu prototype
            for i, batch in enumerate(dataloader):
                batch = batch.cuda().to_dict()
                outputs = self.gather_outputs(model.prompt_model(batch))

                label_batch = batch['label']
                batch_size = len(label_batch)

                hidden = outputs[0][:batch_size, -1, :].to(self.head.weight.dtype)
                logits = outputs[1][:batch_size, -1, :].to(self.head.weight.dtype)

                processed_logits = self.process_logits(logits)
                total_num += batch_size
                all_embeds.append(hidden)
                all_labels.append(label_batch)
                all_model_logits.append(processed_logits)
                
        if all_embeds:
            all_embeds_tensor = torch.cat(all_embeds, dim=0)
            all_labels_tensor = torch.cat(all_labels, dim=0)
            all_model_logits_tensor = torch.cat(all_model_logits, dim=0)
        else:
            print("⚠️ Không có dữ liệu để huấn luyện prototype")
            return

        # Tính bán kính prototype
        dist = []
        for x in embeds:
            if len(x) > 0:  # Bỏ qua các lớp rỗng
                x = torch.stack(x).to(self.head.weight.dtype)
                center = x.mean(0, keepdim=True)
                projected = self.head(x)
                center_proj = self.head(center)
                d = torch.norm(projected - center_proj, dim=-1).mean()
                dist.append(d)
        if dist:
            self.proto_r.data = torch.stack(dist)

        # Huấn luyện prototype
        print(f"🔥 Bắt đầu huấn luyện prototype trong {self.epochs} epoch...")
        for epoch in range(self.epochs):
            # Ánh xạ embedding
            x = self.head(all_embeds_tensor.to(self.head.weight.dtype))
        
            self.optimizer.zero_grad()
            loss = self.loss_func(x, all_model_logits_tensor, all_labels_tensor)
            loss.backward()
            self.optimizer.step()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
            torch.cuda.empty_cache()

        print("Tổng số epoch: {}. Mất mát DecT: {}".format(self.epochs, loss.item()))
        print("Thời gian huấn luyện: {}".format(time.time() - start_time))