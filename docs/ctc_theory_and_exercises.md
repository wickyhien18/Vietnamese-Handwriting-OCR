# Cơ sở Toán học CTC: Chứng minh Quy nạp (Phần B) & Bài tập Số học (Phần C)

## Phần B. Chứng minh công thức truy hồi Forward CTC bằng quy nạp

### 1. Phát biểu bài toán và ký hiệu
Cho chuỗi đặc trưng đầu vào độ dài $T$ với ma trận xác suất khung hình $y_t(k) = P(\pi_t = k \mid X)$ tại mỗi bước thời gian $t \in \{0, 1, \dots, T-1\}$ và ký tự $k \in \Sigma' = \Sigma \cup \{\epsilon\}$ ($\epsilon$ là ký tự `<blank>`).

Gọi $\mathcal{B}: (\Sigma')^T \to \Sigma^{\le T}$ là ánh xạ rút gọn chuỗi đường đi $\pi = (\pi_0, \dots, \pi_{T-1})$ thành chuỗi nhãn bằng hai bước liên tiếp:
1. Gộp các ký tự giống nhau đứng liền kề thành một ký tự duy nhất.
2. Loại bỏ toàn bộ ký tự $\epsilon$.

Với nhãn đích $l = (l_0, l_1, \dots, l_{L-1})$ có độ dài $L$, ta tạo chuỗi mở rộng $l'$ độ dài $U = 2L + 1$ bằng cách xen $\epsilon$ vào đầu, cuối và giữa mọi cặp ký tự:
$$l' = (\epsilon, l_0, \epsilon, l_1, \dots, \epsilon, l_{L-1}, \epsilon)$$
Trong đó $l'(2i) = \epsilon$ với $0 \le i \le L$ và $l'(2i+1) = l_i$ với $0 \le i < L$.

**Định nghĩa biến quy hoạch động $\alpha_t(s)$:**  
Với $0 \le t \le T-1$ và $0 \le s \le 2L$, đặt $\alpha_t(s)$ là tổng xác suất của mọi tiền tố đường đi $\pi_{0:t} = (\pi_0, \dots, \pi_t)$ thỏa mãn:
1. $\pi_t = l'(s)$, và
2. $\mathcal{B}(\pi_{0:t})$ khớp với tiền tố của $l$ tương ứng khi dừng tại vị trí $s$ trên $l'$ (tức là $l'_{0:s}$ sau khi bỏ $\epsilon$).

---

### 2. Chứng minh quy nạp theo bước thời gian $t$

#### Bước cơ sở ($t = 0$)
Tại khung hình đầu tiên $t = 0$, một tiền tố hợp lệ độ dài $1$ chỉ có thể bắt đầu bằng $\epsilon$ (trạng thái $s = 0$) hoặc ký tự đầu tiên $l_0 = l'(1)$ (trạng thái $s = 1$, nếu $L > 0$). Không thể bắt đầu ở $s \ge 2$ vì sẽ bỏ sót ký tự $l_0$. Do đó:
$$\alpha_0(0) = y_0(\epsilon), \qquad \alpha_0(1) = y_0(l'(1)), \qquad \alpha_0(s) = 0 \quad \forall s \ge 2$$
Trong không gian log ($\tilde{\alpha}_t(s) = \ln \alpha_t(s)$), ta thu được đúng điều kiện khởi tạo: $\tilde{\alpha}_0(0) = \ln y_0(\epsilon)$, $\tilde{\alpha}_0(1) = \ln y_0(l'(1))$, và $\tilde{\alpha}_0(s) = -\infty$ với mọi $s \ge 2$.

#### Giả thiết quy nạp
Giả sử tại bước $t - 1$ ($t \ge 1$), $\alpha_{t-1}(s)$ đúng là tổng xác suất của mọi tiền tố đường đi $\pi_{0:t-1}$ kết thúc tại trạng thái $s$ của $l'$ với mọi $s \in \{0, \dots, 2L\}$.

#### Bước quy nạp ($t - 1 \to t$)
Xét một tiền tố $\pi_{0:t} = (\pi_{0:t-1}, l'(s))$ kết thúc tại trạng thái $s$ ở bước $t$. Vì $\mathcal{B}(\pi_{0:t})$ chỉ có thể giữ nguyên hoặc mở rộng thêm tối đa $1$ ký tự của $l'$ so với $\mathcal{B}(\pi_{0:t-1})$, trạng thái tại bước $t-1$ chỉ có thể đến từ **ba nguồn chuyển** hợp lệ:

1. **Giữ nguyên trạng thái ($s \to s$):** $\pi_{t-1} = l'(s)$ và $\pi_t = l'(s)$. Hai khung hình liên tiếp phát ra cùng một token (kể cả $\epsilon$ hay ký tự thường) sẽ được hàm $\mathcal{B}$ gộp lại làm một, nên vị trí trên $l'$ không đổi.
2. **Tiến một bước ($s - 1 \to s$, với $s \ge 1$):** $\pi_{t-1} = l'(s-1)$ và $\pi_t = l'(s)$. Vì $l'$ xen kẽ $\epsilon$ và ký tự nhãn nên $l'(s-1) \neq l'(s)$ luôn đúng; phép co rút $\mathcal{B}$ không gộp hai token này, đưa tiền tố tiến thêm $1$ vị trí trên $l'$.
3. **Nhảy hai bước ($s - 2 \to s$, với $s \ge 2$):** $\pi_{t-1} = l'(s-2)$ và $\pi_t = l'(s)$, tức bỏ qua trạng thái trung gian $s - 1$. Phép chuyển này chỉ hợp lệ khi **đồng thời** thỏa mãn hai điều kiện:
   * **$l'(s) \neq \epsilon$:** Nếu $l'(s) = \epsilon$ thì $l'(s-1)$ bắt buộc là một ký tự nhãn $l_i \neq \epsilon$. Nhảy từ $s-2$ sang $s$ sẽ bỏ qua ký tự nhãn $l_i$, làm sai nhãn đích.
   * **$l'(s) \neq l'(s-2)$:** Khi $l'(s) \neq \epsilon$, vị trí $s-1$ chính là ký tự ngăn cách $\epsilon$ giữa hai ký tự nhãn $l'(s-2)$ và $l'(s)$. Nếu $l'(s) = l'(s-2)$ (ký tự lặp liên tiếp, ví dụ `"oo"` trong `"xoong"`), việc nhảy qua $\epsilon$ ở $s-1$ sẽ khiến $\pi_{t-1} = \pi_t = l'(s)$ và bị hàm $\mathcal{B}$ gộp thành một ký tự duy nhất thay vì hai ký tự. Do đó nhảy hai bước bị cấm khi gặp $\epsilon$ hoặc ký tự lặp.

Vì các tập tiền tố xuất phát từ $s$, $s-1$, $s-2$ tại bước $t-1$ đôi một rời nhau và xác suất phát xạ tại bước $t$ độc lập có điều kiện với quá khứ khi biết trạng thái, theo quy tắc cộng và nhân xác suất:
$$\alpha_t(s) = \left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \mathbb{I}\big[l'(s) \neq \epsilon \land l'(s) \neq l'(s-2)\big]\,\alpha_{t-1}(s-2) \right) \cdot y_t(l'(s))$$
Lấy logarit tự nhiên hai vế cho ra công thức truy hồi trong không gian log:
$$\tilde{\alpha}_t(s) = \operatorname{logsumexp}\Big(\tilde{\alpha}_{t-1}(s),\; \tilde{\alpha}_{t-1}(s-1),\; \tilde{\alpha}_{t-1}(s-2) \text{ nếu hợp lệ}\Big) + \ln y_t(l'(s))$$

#### Kết thúc tại $t = T - 1$
Một đường đi $\pi_{0:T-1}$ có $\mathcal{B}(\pi_{0:T-1}) = l$ khi và chỉ khi nó đi qua toàn bộ các ký tự của $l$ và kết thúc ở trạng thái cuối cùng $s = 2L$ (kết thúc bằng $\epsilon$) hoặc trạng thái áp chót $s = 2L - 1$ (kết thúc bằng ký tự cuối $l_{L-1}$). Vì hai tập đường đi này rời nhau:
$$P(l \mid X) = \sum_{\pi:\,\mathcal{B}(\pi)=l} P(\pi \mid X) = \alpha_{T-1}(2L) + \alpha_{T-1}(2L - 1)$$
Suy ra hàm mất mát CTC là:
$$\mathcal{L}_{\text{CTC}}(X, l) = -\ln P(l \mid X) = -\operatorname{logsumexp}\big(\tilde{\alpha}_{T-1}(2L),\; \tilde{\alpha}_{T-1}(2L - 1)\big) \qquad \blacksquare$$

*(Kiểm chứng thực nghiệm bằng vét cạn toàn bộ $C^T$ đường đi được cài đặt tại `tests/test_ctc_loss.py::test_exhaustive_enumeration_matches_forward_dp`.)*

---

## Phần C. Bài tập số học tính tay có đáp án

### Bài tập 1: Nhãn đơn ký tự ($T = 3$, $l = \text{"a"}$)

**Đề bài:**  
Cho bảng chữ cái $\Sigma' = \{\epsilon, \text{a}\}$ (trong đó $\epsilon$ ở chỉ số $0$, $\text{a}$ ở chỉ số $1$), số khung thời gian $T = 3$, và nhãn đích $l = \text{"a"}$ ($L = 1$). Ma trận xác suất đầu ra $y_t(k)$ của mô hình tại $3$ khung hình được cho như sau:

| Khung $t$ | $y_t(\epsilon)$ | $y_t(\text{a})$ |
| :---: | :---: | :---: |
| $t = 0$ | $0{,}6$ | $0{,}4$ |
| $t = 1$ | $0{,}2$ | $0{,}8$ |
| $t = 2$ | $0{,}5$ | $0{,}5$ |

Hãy tính bảng biến quy hoạch động $\alpha_t(s)$ theo từng bước thời gian và tính giá trị hàm mất mát $\mathcal{L}_{\text{CTC}}$.

**Lời giải chi tiết:**

1. **Mở rộng nhãn:** Với $l = (\text{a})$, chuỗi mở rộng độ dài $2L + 1 = 3$ là $l' = (\epsilon, \text{a}, \epsilon)$ ứng với $s \in \{0, 1, 2\}$.
2. **Khởi tạo ($t = 0$):**
   * $\alpha_0(0) = y_0(\epsilon) = 0{,}6$
   * $\alpha_0(1) = y_0(\text{a}) = 0{,}4$
   * $\alpha_0(2) = 0$
3. **Truy hồi tại $t = 1$:**
   * $s = 0$ ($l'_0 = \epsilon$): $\alpha_1(0) = \alpha_0(0) \cdot y_1(\epsilon) = 0{,}6 \times 0{,}2 = 0{,}12$
   * $s = 1$ ($l'_1 = \text{a}$): $\alpha_1(1) = \big(\alpha_0(1) + \alpha_0(0)\big) \cdot y_1(\text{a}) = (0{,}4 + 0{,}6) \times 0{,}8 = 0{,}80$
   * $s = 2$ ($l'_2 = \epsilon$, cấm nhảy từ $s=0$ vì $l'_2 = \epsilon$):  
     $\alpha_1(2) = \big(\alpha_0(2) + \alpha_0(1)\big) \cdot y_1(\epsilon) = (0 + 0{,}4) \times 0{,}2 = 0{,}08$
4. **Truy hồi tại $t = 2$:**
   * $s = 0$ ($l'_0 = \epsilon$): $\alpha_2(0) = \alpha_1(0) \cdot y_2(\epsilon) = 0{,}12 \times 0{,}5 = 0{,}06$
   * $s = 1$ ($l'_1 = \text{a}$): $\alpha_2(1) = \big(\alpha_1(1) + \alpha_1(0)\big) \cdot y_2(\text{a}) = (0{,}80 + 0{,}12) \times 0{,}5 = 0{,}46$
   * $s = 2$ ($l'_2 = \epsilon$, cấm nhảy từ $s=0$):  
     $\alpha_2(2) = \big(\alpha_1(2) + \alpha_1(1)\big) \cdot y_2(\epsilon) = (0{,}08 + 0{,}80) \times 0{,}5 = 0{,}44$

Bảng tổng hợp $\alpha_t(s)$:

| Trạng thái $s$ ($l'_s$) | $t = 0$ | $t = 1$ | $t = 2$ |
| :---: | :---: | :---: | :---: |
| $s = 0$ ($\epsilon$) | $0{,}60$ | $0{,}12$ | $0{,}06$ |
| $s = 1$ ($\text{a}$) | $0{,}40$ | $0{,}80$ | $0{,}46$ |
| $s = 2$ ($\epsilon$) | $0{,}00$ | $0{,}08$ | $0{,}44$ |

5. **Kết thúc và đối chiếu vét cạn:**
   * Tổng xác suất các đường đi co về `"a"`:
     $$P(\text{"a"} \mid X) = \alpha_2(2) + \alpha_2(1) = 0{,}44 + 0{,}46 = 0{,}90$$
   * *(Kiểm tra bằng liệt kê toàn bộ $2^3 = 8$ đường đi: chỉ có $(\epsilon,\epsilon,\epsilon)$ cho rỗng với xác suất $0{,}06$ và $(\text{a},\epsilon,\text{a})$ cho `"aa"` với xác suất $0{,}04$; cả $6$ đường đi còn lại đều co về `"a"` với tổng xác suất $1 - 0{,}06 - 0{,}04 = 0{,}90$.)*
   * **CTC Loss:**
     $$\mathcal{L}_{\text{CTC}} = -\ln(0{,}90) \approx 0{,}105361$$

---

### Bài tập 2: Nhãn có ký tự lặp ($l = \text{"aa"}$, so sánh $T = 2$ và $T = 3$)

**Đề bài:**  
Giữ nguyên bảng chữ cái $\Sigma' = \{\epsilon, \text{a}\}$ và bảng xác suất ở Bài tập 1, nhưng đổi nhãn đích thành $l = \text{"aa"}$ ($L = 2$).
1. Giải thích vì sao khi $T = 2$ thì không tồn tại căn chỉnh hợp lệ ($\mathcal{L}_{\text{CTC}} = +\infty$).
2. Tính $\alpha_t(s)$ và $\mathcal{L}_{\text{CTC}}$ khi $T = 3$.

**Lời giải chi tiết:**

Chuỗi mở rộng của $l = (\text{a}, \text{a})$ có độ dài $2L + 1 = 5$ là:
$$l' = (\epsilon,\; \text{a},\; \epsilon,\; \text{a},\; \epsilon) \quad (s \in \{0, 1, 2, 3, 4\})$$

**1. Trường hợp $T = 2$ ($t \in \{0, 1\}$):**
* Tại $t = 0$, chỉ có $\alpha_0(0) = 0{,}6$ và $\alpha_0(1) = 0{,}4$ khác $0$; các trạng thái $s \ge 2$ đều bằng $0$.
* Tại $t = 1$, xét trạng thái $s = 3$ ($l'_3 = \text{a}$): theo quy tắc truy hồi, bước nhảy $s - 2 = 1 \to 3$ **bị cấm** vì $l'_3 = \text{a} = l'_1$. Do đó $\alpha_1(3)$ chỉ nhận đóng góp từ $\alpha_0(3) = 0$ và $\alpha_0(2) = 0$, suy ra $\alpha_1(3) = 0$ và $\alpha_1(4) = 0$.
* Tại bước kết thúc $t = 1$: $P(\text{"aa"} \mid X) = \alpha_1(4) + \alpha_1(3) = 0 \implies \mathcal{L}_{\text{CTC}} = -\ln(0) = +\infty$.  
* **Ý nghĩa:** Vì giữa hai ký tự `'a'` giống nhau bắt buộc phải có ít nhất $1$ khung hình phát ra $\epsilon$ để ngăn cách, độ dài tối thiểu của đầu vào phải là $T \ge L + 1 = 3$.

**2. Trường hợp $T = 3$ ($t \in \{0, 1, 2\}$):**
* **$t = 0$:** $\alpha_0(0) = 0{,}6$, $\alpha_0(1) = 0{,}4$, $\alpha_0(2) = \alpha_0(3) = \alpha_0(4) = 0$.
* **$t = 1$:**
  * $\alpha_1(0) = 0{,}12$, $\alpha_1(1) = 0{,}80$, $\alpha_1(2) = 0{,}08$ (tính giống Bài 1).
  * $\alpha_1(3) = \big(\alpha_0(3) + \alpha_0(2)\big) \cdot y_1(\text{a}) = (0 + 0) \times 0{,}8 = 0$ (cấm nhảy từ $s = 1$).
  * $\alpha_1(4) = 0$.
* **$t = 2$:**
  * $\alpha_2(3) = \big(\alpha_1(3) + \alpha_1(2)\big) \cdot y_2(\text{a}) = (0 + 0{,}08) \times 0{,}5 = 0{,}04$ (cấm nhảy từ $s = 1$).
  * $\alpha_2(4) = \big(\alpha_1(4) + \alpha_1(3)\big) \cdot y_2(\epsilon) = (0 + 0) \times 0{,}5 = 0$ (cấm nhảy từ $s = 2$ vì $l'_4 = \epsilon$).
* **Kết thúc tại $t = 2$:**
  * $P(\text{"aa"} \mid X) = \alpha_2(4) + \alpha_2(3) = 0 + 0{,}04 = 0{,}04$ (khớp duy nhất với đường đi $(\text{a}, \epsilon, \text{a})$ có xác suất $0{,}4 \times 0{,}2 \times 0{,}5 = 0{,}04$).
  * **CTC Loss:**
    $$\mathcal{L}_{\text{CTC}} = -\ln(0{,}04) \approx 3{,}218876$$