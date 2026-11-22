# Vai B: chứng minh (Phần B) và bài tập (Phần C)

Tài liệu này chứng minh ba mệnh đề mà code của vai B dựa vào, mỗi mệnh đề kèm một bài tập tính tay có đáp án. Ký hiệu khớp với code:
- `src/corrector/viterbi.py`: Viterbi
- `src/corrector/ngram.py`: n-gram và làm mịn
- `src/corrector/calibration.py`: hiệu chỉnh

Các mệnh đề của Dương (truy hồi forward của CTC) và của Hiển (quy hoạch động Levenshtein, CER/WER, bootstrap, ngưỡng Bayes, cận Hoeffding) không lặp lại ở đây.

## Danh sách mệnh đề

| # | Mệnh đề | Code | Bài tập |
|---|---|---|---|
| 1 | Viterbi trả về chuỗi có điểm cao nhất trong mọi chuỗi ứng viên | `viterbi()` | C1 |
| 2a | Ước lượng hợp lý cực đại của bigram là tỉ số đếm | `NgramModel.train`, `p_bigram` | C2 |
| 2b | Làm mịn add-one và nội suy tuyến tính vẫn cho tổng xác suất bằng 1 | `p_unigram`, `p_bigram` | C2 |
| 3a | Chia logit cho $T > 0$ không đổi dự đoán, nên không đổi độ chính xác | `calibration.py` | C3 |
| 3b | NLL là hàm lồi theo $\beta = 1/T$, nên tìm kiếm tam phân tìm được cực tiểu toàn cục | `fit_temperature` | C3 |
| 3c | Với một nhóm dự đoán giống hệt nhau, $T$ tối ưu làm độ tin cậy bằng đúng tỉ lệ đúng (ECE = 0) | `fit_temperature`, `expected_calibration_error` | C3 |

---

## Mệnh đề 1. Tính tối ưu của Viterbi cho mô hình bigram

**Thiết lập.** Ta có câu quan sát $o_1, \dots, o_n$ (âm tiết OCR). Mỗi vị trí $i$ có một tập ứng viên $C_i$, là các dạng có dấu của $o_i$ (`model.candidates(o_i, k)`). Một chuỗi ứng viên $w = (w_1, \dots, w_n)$ với $w_i \in C_i$ có điểm

$$
S(w) = \sum_{i=1}^{n} \Big[ \log P(w_i \mid w_{i-1}) + \log e(o_i \mid w_i) \Big] + \log P(\langle/s\rangle \mid w_n), \qquad w_0 = \langle s\rangle .
$$

Đây đúng là hàm `path_log_prob`. $P(\cdot\mid\cdot)$ là xác suất bigram đã làm mịn, còn $e$ là xác suất phát xạ (`Emission`).

**Định nghĩa.** Với $1 \le i \le n$ và $c \in C_i$, gọi $\delta_i(c)$ là điểm lớn nhất của một tiền tố dài $i$ kết thúc bằng $c$:

$$
\delta_i(c) = \max_{w_1, \dots, w_{i-1}} \; \sum_{j=1}^{i} \Big[ \log P(w_j \mid w_{j-1}) + \log e(o_j \mid w_j) \Big], \qquad w_i = c .
$$

**Mệnh đề.** Với mọi $i \ge 2$ và $c \in C_i$:

$$
\delta_1(c) = \log P(c \mid \langle s\rangle) + \log e(o_1 \mid c), \qquad
\delta_i(c) = \max_{p \in C_{i-1}} \big[ \delta_{i-1}(p) + \log P(c \mid p) \big] + \log e(o_i \mid c),
$$

và điểm tốt nhất của cả câu là $\max_w S(w) = \max_{c \in C_n} \big[ \delta_n(c) + \log P(\langle/s\rangle \mid c) \big]$. Lần ngược các con trỏ `back` từ ứng viên tốt nhất ở vị trí $n$ cho ra một chuỗi đạt điểm đó.

**Chứng minh (quy nạp theo $i$).**

*Cơ sở $i = 1$.* Tiền tố chỉ có $w_1 = c$, nên tổng có đúng một số hạng, đúng với công thức.

*Bước quy nạp.* Giả sử công thức đúng cho $i - 1$. Tách số hạng cuối của một tiền tố dài $i$ kết thúc bằng $c$, với $p = w_{i-1}$:

$$
\underbrace{\sum_{j=1}^{i-1} [\dots]}_{\text{tiền tố dài } i-1 \text{ kết thúc bằng } p} \; + \; \log P(c \mid p) + \log e(o_i \mid c).
$$

Hai số hạng sau chỉ phụ thuộc $p$ và $c$, không phụ thuộc $w_1, \dots, w_{i-2}$. Vì vậy, khi $p$ cố định, tổng lớn nhất khi phần đầu lớn nhất, tức là bằng $\delta_{i-1}(p)$. Đây là **cấu trúc con tối ưu**: phần đầu của một đường đi tốt nhất kết thúc bằng $(p, c)$ phải là đường đi tốt nhất kết thúc bằng $p$. Nếu không, ta thay phần đầu bằng một đường tốt hơn và được tổng lớn hơn, mâu thuẫn. Lấy max theo $p$ cho ra đúng công thức.

*Kết thúc câu.* $S(w) = (\text{tổng đến vị trí } n) + \log P(\langle/s\rangle \mid w_n)$. Số hạng cuối chỉ phụ thuộc $w_n$, nên lập luận trên cho $\max_w S(w) = \max_c [\delta_n(c) + \log P(\langle/s\rangle \mid c)]$.

*Lần ngược.* `back[i][c]` lưu một $p$ đạt max ở bước $i$. Đi ngược từ ứng viên tốt nhất ở vị trí $n$ ta được từng tiền tố tối ưu, nên chuỗi thu được có điểm đúng bằng $\max_w S(w)$. Khi có nhiều $p$ cùng đạt max, chọn $p$ nào cũng được một chuỗi tối ưu. $\blacksquare$

**Chi phí.** Mỗi vị trí cần $|C_{i-1}| \cdot |C_i| \le K^2$ phép cộng, tổng cộng $O(nK^2)$. Thử hết mọi chuỗi cần $\prod_i |C_i| \le K^n$ phép tính.

**Phạm vi.** Mệnh đề nói về chuỗi tốt nhất **trong các ứng viên** (tối đa $k = 8$ dạng thường gặp nhất, cộng với chính dạng quan sát nếu kho có dạng đó). Nếu dạng đúng không nằm trong $C_i$ thì không chuỗi nào có thể đúng ở vị trí đó. Việc này đã được đo trong `reference_check.compare_viterbi_brute_force` (so với vét cạn trên câu từ 6 âm tiết trở xuống) và `compare_viterbi_hmmlearn`.

---

## Mệnh đề 2a. Ước lượng hợp lý cực đại của bigram bằng tần suất

**Thiết lập.** Với mỗi từ đứng trước $h$ (gồm cả $\langle s\rangle$) và mỗi từ đứng sau $w$ (gồm cả $\langle/s\rangle$), gọi $c(h, w)$ là số lần $w$ đứng ngay sau $h$ trong kho. Tham số là $\theta_{w \mid h} \ge 0$ với $\sum_w \theta_{w \mid h} = 1$ cho mỗi $h$. Log-hợp lý của kho:

$$
\mathcal{L}(\theta) = \sum_h \sum_w c(h, w) \log \theta_{w \mid h} .
$$

**Mệnh đề.** $\mathcal{L}$ đạt cực đại tại $\hat\theta_{w \mid h} = \dfrac{c(h, w)}{c(h)}$ với $c(h) = \sum_w c(h, w)$.

**Chứng minh.** Các ràng buộc tách rời theo $h$, nên ta cực đại hóa riêng từng $h$: $\mathcal{L}_h = \sum_w c(h,w) \log \theta_{w\mid h}$ với $\sum_w \theta_{w\mid h} = 1$. Lập hàm Lagrange

$$
\Lambda = \sum_w c(h, w) \log \theta_{w \mid h} - \lambda \Big( \sum_w \theta_{w \mid h} - 1 \Big).
$$

Đạo hàm theo $\theta_{w\mid h}$ bằng 0 cho $c(h, w) / \theta_{w \mid h} = \lambda$, tức là $\theta_{w \mid h} = c(h, w) / \lambda$. Cộng theo $w$ và dùng ràng buộc: $1 = c(h)/\lambda$, nên $\lambda = c(h)$ và $\hat\theta_{w\mid h} = c(h,w)/c(h)$.

Đây là cực đại chứ không phải cực tiểu: $\mathcal{L}_h$ là tổng các hàm $\log$ lõm nhân hệ số không âm, nên lõm; tập ràng buộc là đơn hình, một tập lồi. Một điểm dừng của hàm lõm trên tập lồi là cực đại toàn cục. Nếu $c(h, w) = 0$ thì $\hat\theta = 0$ cũng là cực đại, vì số hạng đó không xuất hiện trong $\mathcal{L}_h$, và dồn khối xác suất sang các từ khác chỉ làm $\mathcal{L}_h$ tăng. $\blacksquare$

**Khớp với code.** `p_bigram` dùng mẫu số `unigram[prev]` thay cho $\sum_w c(h, w)$. Hai số này bằng nhau vì mỗi lần $h$ xuất hiện, đứng ngay sau nó có đúng một token: một từ, hoặc $\langle/s\rangle$ nếu $h$ ở cuối câu (`train` đếm cả cặp $(w_n, \langle/s\rangle)$). Với $h = \langle s\rangle$, `unigram["<s>"]` là số câu, cũng chính là số cặp $(\langle s\rangle, w_1)$.

## Mệnh đề 2b. Làm mịn vẫn cho tổng bằng 1

**Từ vựng dự đoán.** $V = W \cup \{\langle/s\rangle, \langle unk\rangle\}$, trong đó $W$ là tập các từ trong kho, nên $|V| = |W| + 2$. Trong code, `len(unigram) + 1` bằng đúng $|W| + 2$, vì `unigram` chứa $W$ cộng thêm khóa `<s>`. Gọi $N = \sum_{w \in W} c(w)$ là số từ (`total_words`). Hai từ $\langle/s\rangle$ và $\langle unk\rangle$ có $c = 0$ trong `unigram`.

**Add-one (`p_unigram`):** $P_1(w) = \dfrac{c(w) + 1}{N + |V|}$. Ta có

$$
\sum_{w \in V} P_1(w) = \frac{\sum_{w \in V} c(w) + |V|}{N + |V|} = \frac{N + |V|}{N + |V|} = 1 .
$$

**Nội suy (`p_bigram`):** $P(w \mid h) = \lambda \hat\theta_{w \mid h} + (1 - \lambda) P_1(w)$ với $0 \le \lambda \le 1$. Do $\hat\theta_{\cdot\mid h}$ có tổng bằng 1 trên $V$ (theo 2a; $\hat\theta_{\langle unk\rangle \mid h} = 0$), ta có

$$
\sum_{w \in V} P(w \mid h) = \lambda \cdot 1 + (1 - \lambda) \cdot 1 = 1 ,
$$

và $P(w \mid h) \ge (1-\lambda)/(N + |V|) > 0$ khi $\lambda < 1$, nên $\log$ không bao giờ gặp số 0. $\blacksquare$

**Lưu ý trung thực.** Mô hình đóng gói còn chạy thêm `prune(min_count=2)`, tức là bỏ các cặp bigram chỉ gặp một lần. Khi đó $\sum_w \hat\theta_{w \mid h} \le 1$ và tổng chỉ còn $\le 1$: mô hình bị **thiếu khối** (deficient), phần khối bị bỏ không được chia lại. Điều này không ảnh hưởng tới Viterbi (Mệnh đề 1 chỉ cần so sánh điểm) và độ tin cậy (softmax chuẩn hóa lại trên ứng viên). Nhưng không nên gọi các số đó là phân phối xác suất chính xác. Đối chiếu với `nltk.lm` (mục 7) được làm trên mô hình **không** cắt tỉa.

---

## Mệnh đề 3. Hiệu chỉnh bằng temperature scaling

`calibration.py` dùng đúng một phương pháp: **temperature scaling**. Với mỗi âm tiết có $m \ge 2$ ứng viên và điểm (logit) $z \in \mathbb{R}^m$, độ tin cậy là $\mathrm{softmax}(z / T)$. Tham số $T$ được chọn trên val để cực tiểu NLL, bằng tìm kiếm tam phân trên $\beta = 1/T$ (`fit_temperature`). ECE (`expected_calibration_error`) chia $[0, 1]$ thành 10 khoảng bằng nhau, trọng số mỗi khoảng là tỉ lệ số mẫu rơi vào khoảng đó:

$$
\mathrm{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{N} \big| \mathrm{acc}(B_b) - \mathrm{conf}(B_b) \big| .
$$

**3a. Không đổi dự đoán.** Với $T > 0$, phép $z \mapsto z/T$ giữ nguyên thứ tự các thành phần, và $\mathrm{softmax}$ đơn điệu tăng theo từng thành phần. Vì vậy $\arg\max_j \mathrm{softmax}(z/T)_j = \arg\max_j z_j$: dự đoán, và do đó độ chính xác, không đổi. Chỉ độ tin cậy thay đổi. $\blacksquare$

**3b. NLL lồi theo $\beta$.** Với $N$ mẫu, nhãn đúng $y_n$:

$$
\mathrm{NLL}(\beta) = \frac{1}{N} \sum_{n=1}^{N} \Big[ \log \sum_{j} e^{\beta z_{n j}} - \beta z_{n y_n} \Big].
$$

Đặt $f_n(\beta) = \log \sum_j e^{\beta z_{nj}}$. Đạo hàm bậc một là $f_n'(\beta) = \sum_j p_j z_{nj}$, với $p = \mathrm{softmax}(\beta z_n)$. Đạo hàm bậc hai là $f_n''(\beta) = \sum_j p_j z_{nj}^2 - \big(\sum_j p_j z_{nj}\big)^2 = \mathrm{Var}_{p}(z_n) \ge 0$. Vậy $f_n$ lồi; $-\beta z_{ny_n}$ tuyến tính; tổng các hàm lồi là hàm lồi. Một hàm lồi trên đoạn $[\beta_{\min}, \beta_{\max}]$ thì giảm rồi tăng, không có hai "thung lũng". Tìm kiếm tam phân so sánh hai điểm $m_1 < m_2$:
- nếu $\mathrm{NLL}(m_1) < \mathrm{NLL}(m_2)$, cực tiểu không thể nằm bên phải $m_2$;
- ngược lại, cực tiểu không thể nằm bên trái $m_1$.

Mỗi bước giữ lại $2/3$ đoạn, sau 80 bước còn $(2/3)^{80} \approx 10^{-14}$ độ dài ban đầu. Code tìm trong $T \in [0.05, 20]$: nếu cực tiểu nằm ngoài đoạn này, kết quả là đầu mút gần nhất. $\blacksquare$

**3c. Nhóm dự đoán giống nhau được hiệu chỉnh hoàn hảo.** Giả sử $N$ mẫu đều có cùng logit $z = (a, 0)$ với $a > 0$, nên luôn dự đoán ứng viên 1, và đúng ở $k$ mẫu, với $0 < k < N$. Khi đó, với $s(\beta) = \sigma(a\beta) = 1/(1 + e^{-a\beta})$:

$$
\mathrm{NLL}(\beta) = -\frac{1}{N} \big[ k \log s + (N - k) \log(1 - s) \big].
$$

Vì $ds/d\beta = a\,s(1-s)$, ta có $\dfrac{d\,\mathrm{NLL}}{d\beta} = -\dfrac{a}{N} \big[ k(1-s) - (N-k)s \big] = -\dfrac{a}{N}(k - N s)$. Đạo hàm bằng 0 khi và chỉ khi $s = k/N$. Theo 3b đây là cực tiểu duy nhất. Tại đó độ tin cậy $s$ đúng bằng tỉ lệ đúng $k/N$. Tất cả mẫu rơi vào cùng một khoảng của ECE, nên ECE $= |k/N - s| = 0$. $\blacksquare$

Ý nghĩa: temperature scaling kéo độ tin cậy trung bình về gần tỉ lệ đúng. Nó không sửa được trường hợp mô hình tự tin sai **khác nhau** giữa các nhóm, vì chỉ có một tham số $T$ chung.

---

## Phần C. Bài tập tính tay (có đáp án)

### C1. Viterbi so với vét cạn (Mệnh đề 1)

OCR đọc ra "ban ghe moi" (mất hết dấu). Ứng viên: $C_1 = \{$bạn, bàn$\}$, $C_2 = \{$ghế, ghe$\}$, $C_3 = \{$mới, mời$\}$. Phát xạ: $e(o \mid c) = 0{,}7$ nếu $c$ trùng với chữ quan sát, ngược lại $0{,}3$ (chỉ "ghe" trùng).

| Chuyển | Xác suất | Chuyển | Xác suất |
|---|---|---|---|
| $P(\text{bạn}\mid\langle s\rangle)$ | 0,4 | $P(\text{bàn}\mid\langle s\rangle)$ | 0,1 |
| $P(\text{ghế}\mid\text{bạn})$ | 0,05 | $P(\text{ghe}\mid\text{bạn})$ | 0,10 |
| $P(\text{ghế}\mid\text{bàn})$ | 0,60 | $P(\text{ghe}\mid\text{bàn})$ | 0,05 |
| $P(\text{mới}\mid\text{ghế})$ | 0,50 | $P(\text{mời}\mid\text{ghế})$ | 0,10 |
| $P(\text{mới}\mid\text{ghe})$ | 0,20 | $P(\text{mời}\mid\text{ghe})$ | 0,30 |
| $P(\langle/s\rangle\mid\text{mới})$ | 0,5 | $P(\langle/s\rangle\mid\text{mời})$ | 0,4 |

**a)** Tính xác suất của cả 8 chuỗi bằng vét cạn. **b)** Lập bảng $\delta$ của Viterbi (dùng tích xác suất thay cho tổng log) và lần ngược. **c)** Nếu chọn tham lam ứng viên tốt nhất ở từng vị trí thì được chuỗi nào? Vì sao sai?

**Đáp án.**

a) Ví dụ: $P(\text{bàn ghế mới}) = 0{,}1 \cdot 0{,}3 \cdot 0{,}6 \cdot 0{,}3 \cdot 0{,}5 \cdot 0{,}3 \cdot 0{,}5 = 0{,}000405$.

| Chuỗi | Xác suất | Chuỗi | Xác suất |
|---|---|---|---|
| bạn ghế mới | 0,0001350 | bàn ghế mới | **0,0004050** |
| bạn ghế mời | 0,0000216 | bàn ghế mời | 0,0000648 |
| bạn ghe mới | 0,0002520 | bàn ghe mới | 0,0000315 |
| bạn ghe mời | 0,0003024 | bàn ghe mời | 0,0000378 |

Chuỗi tốt nhất là **bàn ghế mới**.

b) Bảng $\delta$:
- $\delta_1$: bạn $= 0{,}4 \cdot 0{,}3 = 0{,}12$; bàn $= 0{,}1 \cdot 0{,}3 = 0{,}03$.
- $\delta_2$: ghế $= \max(0{,}12 \cdot 0{,}05;\ 0{,}03 \cdot 0{,}6) \cdot 0{,}3 = 0{,}018 \cdot 0{,}3 = 0{,}0054$ (từ bàn); ghe $= \max(0{,}12 \cdot 0{,}1;\ 0{,}03 \cdot 0{,}05) \cdot 0{,}7 = 0{,}0084$ (từ bạn).
- $\delta_3$: mới $= \max(0{,}0054 \cdot 0{,}5;\ 0{,}0084 \cdot 0{,}2) \cdot 0{,}3 = 0{,}00081$ (từ ghế); mời $= \max(0{,}0054 \cdot 0{,}1;\ 0{,}0084 \cdot 0{,}3) \cdot 0{,}3 = 0{,}000756$ (từ ghe).
- Kết thúc câu: mới $0{,}00081 \cdot 0{,}5 = 0{,}000405$; mời $0{,}000756 \cdot 0{,}4 = 0{,}0003024$. Chọn **mới**.
- Lần ngược: mới ← ghế ← bàn. Kết quả **bàn ghế mới** với $0{,}000405$, đúng bằng kết quả vét cạn.

Viterbi chỉ cần $2 + 4 + 4 + 2 = 12$ phép nhân-so sánh, thay vì tính đủ 8 chuỗi.

c) Chọn tham lam: vị trí 1 lấy bạn ($0{,}12 > 0{,}03$), vị trí 2 lấy ghe ($0{,}0084 > 0{,}0054$). Kết quả "bạn ghe mời" (0,0003024) hoặc "bạn ghe mới" (0,000252), đều kém hơn chuỗi tối ưu. Lý do: lựa chọn ở vị trí 1 phải tính đến cả chữ phía sau ($P(\text{ghế}\mid\text{bàn}) = 0{,}6$ rất lớn), và Viterbi giữ lại **mọi** ứng viên ở mỗi vị trí cho tới cuối câu.

### C2. Tổng xác suất bằng 1 (Mệnh đề 2)

Kho có hai câu: "ta đi", "ta về". Cho $\lambda = 0{,}8$.

**a)** Tính $W$, $N$, $|V|$ và xác suất add-one của mọi từ trong $V$. **b)** Tính $\hat\theta_{\cdot\mid\text{ta}}$ bằng tỉ số đếm và kiểm tra rằng đây là nghiệm Lagrange. **c)** Tính $P(w \mid \text{ta})$ đã nội suy cho mọi $w \in V$ và kiểm tra tổng.

**Đáp án.**

a) $W = \{$ta, đi, về$\}$, $N = 4$, $|V| = 3 + 2 = 5$. Từ đó $P_1(\text{ta}) = 3/9$, $P_1(\text{đi}) = P_1(\text{về}) = 2/9$, $P_1(\langle/s\rangle) = P_1(\langle unk\rangle) = 1/9$. Tổng $9/9 = 1$.

b) $c(\text{ta}) = 2$, $c(\text{ta}, \text{đi}) = c(\text{ta}, \text{về}) = 1$, nên $\hat\theta_{\text{đi}\mid\text{ta}} = \hat\theta_{\text{về}\mid\text{ta}} = 1/2$, các từ khác bằng 0. Kiểm tra: với $\lambda_{\text{Lagrange}} = c(\text{ta}) = 2$ ta có $c(\text{ta}, \text{đi})/\theta = 1/(1/2) = 2 = \lambda$. Mẫu số `unigram["ta"]` $= 2$ đúng bằng tổng hàng $1 + 1$, vì sau mỗi chữ "ta" có đúng một chữ.

c) Với $\lambda = 0{,}8$:
- đi và về: $0{,}8 \cdot \tfrac12 + 0{,}2 \cdot \tfrac29 = 0{,}4444$;
- ta: $0{,}2 \cdot \tfrac39 = 0{,}0667$;
- $\langle/s\rangle$ và $\langle unk\rangle$: $0{,}2 \cdot \tfrac19 = 0{,}0222$.

Tổng $0{,}4444 \cdot 2 + 0{,}0667 + 0{,}0222 \cdot 2 = 1{,}0000$.

### C3. ECE trước và sau hiệu chỉnh (Mệnh đề 3)

Bốn âm tiết, mỗi âm tiết có hai ứng viên với cùng logit $z = (3, 0)$. Mô hình luôn chọn ứng viên 1 và đúng ở 3 trong 4 âm tiết.

**a)** Tính độ chính xác, độ tin cậy và ECE khi $T = 1$. **b)** Làm lại với $T = 2$. **c)** Tìm $T$ tối ưu theo NLL. Tính ECE tại đó. **d)** Độ chính xác có thay đổi theo $T$ không?

**Đáp án.**

a) Độ chính xác $= 3/4 = 0{,}75$. Độ tin cậy $= \sigma(3) = 0{,}9526$. Cả 4 mẫu nằm trong khoảng $(0{,}9;\ 1]$, nên ECE $= |0{,}75 - 0{,}9526| = 0{,}2026$: mô hình quá tự tin. NLL $= 0{,}7986$.

b) $T = 2$: độ tin cậy $= \sigma(1{,}5) = 0{,}8176$, ECE $= 0{,}0676$, NLL $= 0{,}5764$.

c) Theo 3c, cực tiểu đạt khi $\sigma(3/T) = 3/4$, tức là $3/T = \ln 3$, nên $T = 3/\ln 3 \approx 2{,}731$. Tại đó độ tin cậy $= 0{,}75$, ECE $= 0$, NLL $= 0{,}5623$ (nhỏ hơn hai giá trị ở trên). `fit_temperature` trên dữ liệu này cũng trả về $T \approx 2{,}73$; test `test_overconfident_model_gets_t_above_one` kiểm tra đúng kiểu tình huống này.

d) Không đổi: với mọi $T > 0$, mô hình vẫn chọn ứng viên 1 (Mệnh đề 3a), nên độ chính xác luôn là 0,75.
