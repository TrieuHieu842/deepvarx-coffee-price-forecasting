# CƠ SỞ LÝ THUYẾT VÀ PHƯƠNG PHÁP LUẬN KIỂM ĐỊNH DIEBOLD–MARIANO (DM TEST)
## Bổ trợ Đánh giá So sánh Hiệu suất Mô hình Dự báo Giá Cà phê

**Tác giả:** Triệu Phúc Hiếu, Bùi Quốc Hậu  
**Đề tài:** Tìm hiểu mô hình DeepVARX vào dự báo giá thu mua cà phê ở Việt Nam  
**Tài liệu học thuật chuyên sâu phục vụ nghiên cứu & viết luận văn tốt nghiệp**

---

## 1. Khái niệm và mục đích của kiểm định Diebold–Mariano

Trong kinh tế lượng chuỗi thời gian và khoa học dữ liệu tài chính, việc đánh giá mô hình dự báo thường bắt đầu bằng các chỉ số tổng hợp điểm như RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), hay MAPE (Mean Absolute Percentage Error). Tuy nhiên, các chỉ số này chỉ cung cấp giá trị trung bình mẫu đơn thuần. 

Một mô hình A có RMSE thấp hơn mô hình B trên tập kiểm tra (Test set) chưa chắc đã là mô hình vượt trội thực sự:
1. Sự chênh lệch sai số có thể chỉ là do **biến động ngẫu nhiên (sample noise)** của mẫu dữ liệu hữu hạn.
2. Sai số dự báo giữa hai mô hình có thể có **tương quan chuỗi (autocorrelation)** và **phương sai thay đổi theo thời gian (heteroskedasticity)**.
3. Không thể khẳng định mô hình A tốt hơn B nếu không có một **phép kiểm định giả thuyết thống kê chính thức (formal statistical hypothesis test)** để xác nhận mức ý nghĩa thống kê (statistical significance).

Kiểm định **Diebold–Mariano (DM Test)** ra đời nhằm giải quyết triệt để vấn đề này. Mục đích của kiểm định DM là kiểm tra xem sự khác biệt về độ chính xác dự báo kỳ vọng giữa hai mô hình có khác 0 một cách có ý nghĩa thống kê hay không, dưới một hàm mất mát kinh tế được xác định trước.

---

## 2. Nghiên cứu gốc Diebold & Mariano (1995)

Kiểm định DM được đề xuất bởi **Francis X. Diebold và Roberto S. Mariano** trong bài báo kinh điển:
> *Diebold, F. X., & Mariano, R. S. (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.*

Khác với các kiểm định truyền thống dựa trên phân phối của phần dư mô hình hồi quy (như F-test hoặc Likelihood Ratio test vốn đòi hỏi giả định phân phối chuẩn, mô hình tuyến tính lồng nhau, và phần dư nhiễu trắng), kiểm định Diebold–Mariano có các đặc tính vượt trội:
- **Phi tham số đối với sai số (Distribution-free):** Không đòi hỏi sai số dự báo phải tuân theo phân phối chuẩn, có thể lệch (skewed), có đuôi dày (fat-tailed), hoặc rời rạc.
- **Hỗ trợ hàm mất mát tùy ý (Arbitrary loss functions):** Có thể áp dụng cho hàm mất mát đối xứng (bình phương, tuyệt đối) hoặc phi đối xứng (Linex, Directional loss).
- **Cho phép sai số có tự tương quan (Serial correlation allowed):** Xử lý hoàn hảo hiện tượng tự tương quan bậc $h-1$ vốn dĩ phát sinh trong dự báo đa bước ($h$-step-ahead forecasting).
- **Áp dụng được cho mọi lớp mô hình:** Từ mô hình chuỗi thời gian tuyến tính (ARIMA, VAR, VARX) đến các mô hình phi tuyến và học sâu phức tạp (LSTM, DeepVAR, DeepVARX).

---

## 3. Hệ giả thuyết thống kê $H_0$ và $H_1$

Xét hai mô hình dự báo cạnh tranh: **Mô hình A** và **Mô hình B**.

Gọi chuỗi vi phân mất mát tại thời điểm $t$ là $d_t$. Ta kiểm định hệ giả thuyết:

$$\begin{cases}
H_0: \mathbb{E}[d_t] = 0 & \text{(Hai mô hình có độ chính xác dự báo tương đương về mặt kỳ vọng)} \\
H_1: \mathbb{E}[d_t] \neq 0 & \text{(Hai mô hình có độ chính xác dự báo khác biệt có ý nghĩa thống kê)}
\end{cases}$$

Trong các bài toán thực nghiệm định hướng, ta cũng có thể xem xét kiểm định một phía (one-sided test):
- $H_{1, \text{less}}: \mathbb{E}[d_t] < 0$ (Mô hình A có tổn thất thấp hơn, tức dự báo tốt hơn mô hình B).
- $H_{1, \text{greater}}: \mathbb{E}[d_t] > 0$ (Mô hình B có tổn thất thấp hơn, tức dự báo tốt hơn mô hình A).

Tuy nhiên, theo chuẩn mực nghiên cứu định lượng, kiểm định hai phía (two-sided test) luôn được ưu tiên báo cáo làm cơ sở khách quan.

---

## 4. Công thức sai số dự báo

Giả sử tập kiểm tra có $T$ bước thời gian quan sát: $t = 1, 2, \dots, T$.
Tại mỗi thời điểm $t$, gọi:
- $y_t$: Giá trị thực tế của chuỗi thời gian (ví dụ: giá thu mua cà phê thực tế tại Đắk Lắk).
- $\hat{y}_{A,t}$: Giá trị dự báo của Mô hình A cho thời điểm $t$.
- $\hat{y}_{B,t}$: Giá trị dự báo của Mô hình B cho thời điểm $t$.

Sai số dự báo (forecast errors) của từng mô hình được xác định theo công thức:

$$e_{A,t} = y_t - \hat{y}_{A,t}$$

$$e_{B,t} = y_t - \hat{y}_{B,t}$$

---

## 5. Các hàm mất mát (Loss Functions)

Một ưu điểm lớn của DM Test là cho phép người nghiên cứu đánh giá hiệu suất mô hình dưới các tiêu chuẩn kinh tế khác nhau thông qua hàm mất mát $L(e)$. Phân hệ triển khai các hàm mất mát chuẩn mực sau:

### 5.1. Squared Error Loss (MSE Loss)
$$L_{SE}(e_t) = e_t^2 = (y_t - \hat{y}_t)^2$$
- Đặc điểm: Phạt rất nặng các sai số lớn do tính chất bậc hai. Phù hợp khi chi phí rủi ro của sai số tỷ lệ với bình phương độ lệch giá.
- Trung bình của hàm này tương ứng với chỉ tiêu Mean Squared Error (MSE).

### 5.2. Absolute Error Loss (MAE Loss)
$$L_{AE}(e_t) = |e_t| = |y_t - \hat{y}_t|$$
- Đặc điểm: Đo lường độ lệch tuyến tính, bền vững hơn với các điểm ngoại lai (outliers) hoặc các cú sốc giá cực đoan ngắn hạn trên thị trường cà phê.
- Trung bình của hàm này tương ứng với chỉ tiêu Mean Absolute Error (MAE).

### 5.3. Continuous Ranked Probability Score (CRPS)
Đối với các mô hình dự báo xác suất đa biến (như DeepVAR và DeepVARX), mô hình không chỉ dự báo một điểm kỳ vọng $\hat{y}_t$ mà dự báo toàn bộ phân phối xác suất tích lũy $F_t(y)$. Khi đó, hàm mất mát CRPS được áp dụng:

$$CRPS(F_t, y_t) = \int_{-\infty}^{\infty} [F_t(z) - \mathbb{I}(z \ge y_t)]^2 \, dz$$

Trong trường hợp phân phối dự báo là phân phối chuẩn $\mathcal{N}(\mu_t, \sigma_t^2)$ (hoặc Log-Normal sau chuyển đổi), CRPS có nghiệm giải tích dạng đóng:

$$CRPS(\mu_t, \sigma_t, y_t) = \sigma_t \left[ \frac{y_t - \mu_t}{\sigma_t} \left( 2\Phi\left(\frac{y_t - \mu_t}{\sigma_t}\right) - 1 \right) + 2\phi\left(\frac{y_t - \mu_t}{\sigma_t}\right) - \frac{1}{\sqrt{\pi}} \right]$$

Trong đó $\Phi(\cdot)$ và $\phi(\cdot)$ lần lượt là hàm phân phối tích lũy và hàm mật độ xác suất của phân phối chuẩn tắc $\mathcal{N}(0, 1)$.

---

## 6. Chuỗi vi phân mất mát (Loss Differential Series)

Chuỗi vi phân mất mát tại mỗi thời điểm $t$ được định nghĩa theo đúng thứ tự **Mô hình A − Mô hình B**:

$$d_t = L(e_{A,t}) - L(e_{B,t})$$

### Quy tắc dấu quan trọng:
- Nếu $d_t < 0 \iff L(e_{A,t}) < L(e_{B,t})$: Tại thời điểm $t$, Mô hình A có tổn thất nhỏ hơn Mô hình B (Mô hình A dự báo chính xác hơn).
- Nếu $d_t > 0 \iff L(e_{A,t}) > L(e_{B,t})$: Tại thời điểm $t$, Mô hình B có tổn thất nhỏ hơn Mô hình A (Mô hình B dự báo chính xác hơn).
- Nếu $d_t = 0$: Hai mô hình có độ chính xác hoàn toàn trùng khớp tại thời điểm $t$.

Giá trị vi phân mất mát trung bình mẫu:

$$\bar{d} = \frac{1}{T} \sum_{t=1}^T d_t$$

Nếu $\bar{d} < 0$, mô hình A có tổn thất trung bình thấp hơn mô hình B trên toàn mẫu. Ngược lại, nếu $\bar{d} > 0$, mô hình B có tổn thất trung bình thấp hơn mô hình A.

---

## 7. Công thức thống kê kiểm định Diebold–Mariano

Giả sử chuỗi $d_t$ có tính dừng hiệp phương sai (covariance stationary). Theo Định lý Giới hạn Trung tâm (CLT):

$$\sqrt{T}(\bar{d} - \mu) \xrightarrow{d} \mathcal{N}(0, 2\pi f_d(0))$$

Trong đó $f_d(0)$ là mật độ phổ của $d_t$ tại tần số 0, hay chính là **phương sai tiệm cận dài hạn (long-run variance - LRV)**:

$$\sigma_{LR}^2 = \lim_{T \to \infty} \mathbb{V}\text{ar}(\sqrt{T}\bar{d}) = \sum_{k=-\infty}^{\infty} \gamma_d(k) = \gamma_0 + 2\sum_{k=1}^{\infty} \gamma_k$$

Trong đó $\gamma_k = \text{Cov}(d_t, d_{t-k})$ là hàm tự hiệp phương sai bậc $k$ của $d_t$.

Thống kê Diebold–Mariano tiệm cận được tính bằng:

$$DM = \frac{\bar{d}}{\sqrt{\frac{\hat{\sigma}_{LR}^2}{T}}} = \frac{\bar{d}}{\hat{\text{SE}}(\bar{d})}$$

Dưới giả thuyết $H_0$, thống kê $DM$ hội tụ tiệm cận về phân phối chuẩn tắc:

$$DM \xrightarrow{d} \mathcal{N}(0, 1)$$

---

## 8. Ước lượng Phương sai Dài hạn: Vai trò của Newey–West HAC

### 8.1. Tại sao không thể dùng phương sai mẫu thông thường $s^2 / T$?
Nếu sử dụng công thức phương sai thông thường $s^2 = \frac{1}{T-1}\sum (d_t - \bar{d})^2$, ta đã ngầm giả định các quan sát $d_t$ độc lập với nhau (i.i.d.). Tuy nhiên trong thực tế dự báo chuỗi thời gian:
1. Dự báo $h$ bước ($h \ge 2$) tự động tạo ra cấu trúc tự tương quan bậc ít nhất là $MA(h-1)$ giữa các sai số dự báo chồng lấn.
2. Thị trường nông sản tài chính thường xuyên xuất hiện cụm biến động (volatility clustering) dẫn đến hiện tượng phương sai thay đổi có điều kiện (conditional heteroskedasticity).

Nếu bỏ qua tự tương quan và phương sai thay đổi, ước lượng sai số chuẩn sẽ bị chệch nghiêm trọng (thường bị đánh giá thấp), khiến thống kê DM bị thổi phồng giả tạo và bác bỏ $H_0$ sai lầm (lỗi loại I tăng vọt).

### 8.2. Ước lượng Newey–West (1987) HAC
Để khắc phục, ta sử dụng ước lượng vững với tự tương quan và phương sai thay đổi (Heteroskedasticity and Autocorrelation Consistent - HAC) đề xuất bởi Newey & West (1987):

$$\hat{\sigma}_{HAC}^2 = \hat{\gamma}_0 + 2\sum_{j=1}^{q} w(j, q) \hat{\gamma}_j$$

Trong đó:
- $\hat{\gamma}_j = \frac{1}{T}\sum_{t=j+1}^T (d_t - \bar{d})(d_{t-j} - \bar{d})$ là tự hiệp phương sai mẫu bậc $j$.
- $w(j, q)$ là hàm trọng số hạt nhân Bartlett (Bartlett kernel), đảm bảo ma trận phương sai luôn bán xác định dương ($\hat{\sigma}_{HAC}^2 \ge 0$):
  $$w(j, q) = 1 - \frac{j}{q + 1}$$
- $q$ là độ rộng băng tần (bandwidth hoặc lag truncation parameter).

### 8.3. Nguyên tắc lựa chọn độ rộng trễ $q$ (Bandwidth Selection)
- **Đối với dự báo $h$ bước ($h > 1$):** Về mặt lý thuyết, sai số tối ưu của dự báo $h$ bước là chuỗi $MA(h-1)$. Do đó, trễ tối thiểu bắt buộc phải đặt $q = h - 1$.
- **Đối với dự báo 1 bước ($h = 1$):** Về mặt lý tưởng, sai số 1 bước là nhiễu trắng ($q = 0$). Tuy nhiên, trong thực tế thị trường cà phê, các mô hình học máy phức tạp có thể để lại cấu trúc phụ thuộc phi tuyến hoặc tự tương quan tàn dư trong $d_t$. Do đó, phân hệ áp dụng quy tắc thực nghiệm vững:
  1. Quy tắc Newey–West (1994) / Andrews (1991):
     $$q = \lfloor 4 \cdot (T / 100)^{2/9} \rfloor$$
     Với $T = 542$ ngày test, $q \approx \lfloor 4 \cdot (5.42)^{0.2222} \rfloor = \lfloor 4 \cdot 1.454 \rfloor = 5$.
  2. Cho phép người dùng tùy chỉnh tham số $q$ thông qua tham số dòng lệnh CLI `--hac-lag`.

---

## 9. Hiệu chỉnh mẫu hữu hạn Harvey–Leybourne–Newbold (HLN Correction)

### 9.1. Vấn đề kích thước kiểm định trong mẫu nhỏ
Trong nghiên cứu nổi tiếng:
> *Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281–291.*

Các tác giả đã chứng minh bằng mô phỏng Monte Carlo rằng thống kê DM gốc có xu hướng **bác bỏ $H_0$ quá mức (severely oversized)** khi kích thước mẫu $T$ nhỏ hoặc trung bình ($T < 100$), và độ chệch này càng trầm trọng khi bước dự báo $h$ tăng lên.

### 9.2. Công thức hiệu chỉnh HLN
Harvey, Leybourne và Newbold (1997) đề xuất hệ số hiệu chỉnh mẫu hữu hạn:

$$DM_{HLN} = DM \times \sqrt{\frac{T + 1 - 2h + \frac{h(h - 1)}{T}}{T}}$$

Đặc biệt, khi $h = 1$, số hạng $h(h-1)/T = 0$, công thức trở thành:

$$DM_{HLN} = DM \times \sqrt{\frac{T - 1}{T}}$$

### 9.3. Phân phối tham chiếu Student-$t$
Thay vì so sánh với phân phối chuẩn tắc $\mathcal{N}(0, 1)$, HLN khuyến nghị so sánh $DM_{HLN}$ với phân phối **Student-$t$ với $T - 1$ bậc tự do**:

$$DM_{HLN} \sim t_{(T - 1)}$$

Điều này làm dày đuôi phân phối ở các mẫu nhỏ, kiểm soát chặt chẽ tỷ lệ sai lầm loại I (false positives). Khi $T \to \infty$, hệ số $\sqrt{\frac{T-1}{T}} \to 1$ và phân phối $t_{(T-1)}$ hội tụ về $\mathcal{N}(0, 1)$, đảm bảo tính nhất quán tiệm cận hoàn hảo.

---

## 10. Hiệu chỉnh So sánh Nhiều Mô hình: Holm–Bonferroni Correction

### 10.1. Vấn đề kiểm định đa giả thuyết (Multiple Testing Problem)
Khi nghiên cứu đồng thời nhiều mô hình (ví dụ: so sánh Naive, VAR(5), VARX 2, LSTM 2, DeepVAR 2, DeepVARX 2), ta phải thực hiện đồng thời $m = \binom{K}{2}$ phép kiểm định cặp. 

Nếu mỗi phép kiểm định đều đặt mức ý nghĩa $\alpha = 0.05$, xác suất mắc phải ít nhất một sai lầm loại I trong cả họ kiểm định (Family-Wise Error Rate - FWER) sẽ tăng vọt:

$$FWER \le 1 - (1 - \alpha)^m$$

Ví dụ: Với 6 mô hình ($m = 15$ cặp so sánh), $FWER \le 1 - (0.95)^{15} \approx 53.7\%$. Tức là có hơn $50\%$ khả năng ta kết luận sai rằng có ít nhất một cặp mô hình khác biệt dù thực tế chúng hoàn toàn tương đương!

### 10.2. Thuật toán Holm–Bonferroni (1979)
Phương pháp Holm (Step-down procedure) kiểm soát chặt chẽ FWER ở mức $\alpha$ mà không bị bảo thủ quá mức như Bonferroni truyền thống:

1. Xác định họ phép kiểm định (Family of Tests): Toàn bộ $m$ cặp mô hình so sánh trong cùng một điều kiện cụ thể (ví dụ: cùng 1 tỉnh, cùng 1 horizon $h$, cùng 1 hàm mất mát).
2. Thu thập và sắp xếp các $p$-value gốc theo thứ tự tăng dần:
   $$p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$$
3. Điều chỉnh tuần tự từng $p$-value:
   $$p_{(k)}^{adj} = \min\left( (m - k + 1) \cdot p_{(k)}, \, 1.0 \right), \quad k = 1, \dots, m$$
4. Đảm bảo tính đơn điệu không giảm (Monotonicity enforcement):
   $$p_{(k)}^{final} = \max\left( p_{(k)}^{adj}, \, p_{(k-1)}^{final} \right) \quad \text{với } p_{(0)}^{final} = 0$$
   Quy tắc này bảo đảm: $p_{(1)}^{final} \le p_{(2)}^{final} \le \dots \le p_{(m)}^{final}$ và luôn luôn có $p_{(k)}^{final} \ge p_{(k)}$.

---

## 11. Hướng dẫn đọc Thống kê DM và $p$-value

Khi thứ tự được chuẩn hóa là **Mô hình A − Mô hình B** với $d_t = L(e_{A,t}) - L(e_{B,t})$:

| Giá trị thống kê | Chiều hướng $d_t$ | Ý nghĩa tương đối về tổn thất |
| :---: | :---: | :--- |
| **$DM < 0$** | $\bar{d} < 0 \implies \overline{L(A)} < \overline{L(B)}$ | Mô hình A có tổn thất mẫu **thấp hơn** Mô hình B. |
| **$DM > 0$** | $\bar{d} > 0 \implies \overline{L(A)} > \overline{L(B)}$ | Mô hình A có tổn thất mẫu **cao hơn** Mô hình B (Mô hình B tốt hơn). |
| **$DM = 0$** | $\bar{d} = 0 \implies \overline{L(A)} = \overline{L(B)}$ | Hai mô hình có tổn thất mẫu ngang bằng nhau. |

### Đánh giá ý nghĩa thống kê qua $p$-value (kiểm định 2 phía):
- **$p\text{-value} < \alpha$ (thường là 0.05):** Bác bỏ $H_0$ ở mức tin cậy $1 - \alpha$ (95%).
  - Nếu kết hợp với $DM < 0$: Mô hình A **tốt hơn có ý nghĩa thống kê** so với Mô hình B.
  - Nếu kết hợp với $DM > 0$: Mô hình A **kém hơn có ý nghĩa thống kê** so với Mô hình B (Mô hình B tốt hơn A).
- **$p\text{-value} \ge \alpha$:** Chưa đủ bằng chứng thống kê để bác bỏ $H_0$. Chênh lệch sai số giữa hai mô hình có thể chỉ do nhiễu ngẫu nhiên của mẫu.

---

## 12. Cách diễn giải kết quả: Phân biệt "Hiệu suất mẫu" và "Ý nghĩa thống kê"

> [!IMPORTANT]
> **Quy tắc vàng trong phân tích thực nghiệm:** Tuyệt đối không được đồng nhất *"Mô hình có RMSE/MAE thấp hơn"* với *"Mô hình vượt trội về mặt thống kê"*.

Trong báo cáo khoa học, phân hệ tách bạch rõ ràng hai kết luận:

1. **Better model based on mean loss (Hiệu suất mẫu thuần túy):**
   - Chỉ căn cứ vào dấu của $\bar{d}$: Nếu $\bar{d} < 0 \implies$ Model A; Nếu $\bar{d} > 0 \implies$ Model B.
2. **Final statistical interpretation (Kết luận thống kê cuối cùng):**
   - Phải kết hợp đồng thời giữa dấu của $DM$ và giá trị $p\text{-value}$ (sau điều chỉnh Holm):
     * Nếu $p \ge 0.05$: "Không có sự khác biệt có ý nghĩa thống kê giữa hai mô hình (sự chênh lệch loss chỉ do ngẫu nhiên mẫu)."
     * Nếu $p < 0.05$ và $DM < 0$: "Mô hình A vượt trội hơn Mô hình B có ý nghĩa thống kê ở mức ý nghĩa 5%."
     * Nếu $p < 0.05$ và $DM > 0$: "Mô hình B vượt trội hơn Mô hình A có ý nghĩa thống kê ở mức ý nghĩa 5%."

---

## 13. Các giả định, hạn chế và trường hợp không phù hợp

Để áp dụng DM Test chuẩn mực, nhà nghiên cứu cần nắm vững các giới hạn sau:

### 13.1. Giả định tính dừng (Stationarity of Loss Differential)
Chuỗi vi phân mất mát $d_t$ phải có tính dừng hiệp phương sai. Nếu dữ liệu gốc có xu thế hoặc bước ngẫu nhiên (non-stationary), các sai số dự báo $e_t$ phải được tính trên chuỗi đã chuyển đổi dừng (ví dụ: log-return) hoặc mô hình phải đồng tích hợp. Trong dự án của chúng ta, việc các mô hình dự báo trên chuỗi dừng và quy đổi về VNĐ/kg đã được kiểm định ADF/KPSS cẩn trọng.

### 13.2. Vấn đề mô hình lồng nhau (Nested Models) & Cảnh báo Clark–West (2007)
- **Mô hình lồng nhau là gì?** Khi Mô hình 1 (bị giới hạn - Restricted) là một trường hợp con của Mô hình 2 (không bị giới hạn - Unrestricted). Ví dụ: Mô hình VAR(5) thuần túy lồng bên trong mô hình VARX(5) (vì VARX suy biến thành VAR khi các hệ số ngoại sinh $\beta_X = 0$).
- **Hạn chế của DM:** Clark & McCracken (2001) và Clark & West (2006, 2007) đã chỉ ra rằng khi hai mô hình lồng nhau và $H_0$ đúng, mô hình lớn hơn chứa các tham số dư thừa sẽ bị phạt bởi phương sai ước lượng trong mẫu, dẫn đến phân phối của thống kê DM bị dịch sang trái (non-standard normal). Thống kê DM khi đó bị **thiếu kích thước (undersized)**, dẫn đến việc quá bảo thủ và khó bác bỏ $H_0$.
- **Khuyến nghị xử lý:** Khi so sánh các mô hình lồng nhau (ví dụ VAR vs VARX), kết quả kiểm định DM cần được ghi nhận kèm cảnh báo; trường hợp này có thể sử dụng thêm kiểm định Clark–West (Adjusted MSPE).

### 13.3. Dữ liệu bảng đa tỉnh: Không gộp chuỗi tùy tiện (No Naive Panel Pooling)
Trong bài toán 3 tỉnh Tây Nguyên, giá cà phê tại Đắk Lắk, Lâm Đồng và Gia Lai có tương quan chéo không gian rất cao ($\rho > 0.89$). Nếu gộp thẳng hàng 542 ngày × 3 tỉnh thành một chuỗi duy nhất dài 1,626 quan sát để chạy DM:
- Số quan sát độc lập bị khai khống gấp 3 lần.
- Thống kê DM sẽ bị phóng đại nghiêm trọng, sinh ra $p$-value siêu nhỏ giả tạo.
- **Giải pháp khoa học:** Thực hiện kiểm định DM **độc lập cho từng tỉnh**. Nếu cần một chỉ số toàn vùng Tây Nguyên, phải tính chuỗi trung bình vi phân theo ngày:
  $$\bar{d}_t = \frac{1}{N_{prov}}\sum_{i=1}^{N_{prov}} d_{i,t}$$
  rồi mới thực hiện kiểm định Newey–West HAC trên chuỗi $\bar{d}_t$ đó.

---

## 14. Ứng dụng kiểm định DM trong bài toán dự báo giá cà phê Việt Nam

Trong khuôn khổ đề tài nghiên cứu dự báo giá thu mua cà phê Robusta Việt Nam:
1. **Đối chứng với Naive Random Walk Benchmark:** Kiểm tra xem việc áp dụng các mô hình phức tạp (VARX, LSTM, DeepVAR, DeepVARX) có thực sự mang lại giá trị gia tăng vượt trội hơn quy tắc kinh nghiệm "lấy giá hôm qua làm giá hôm nay" một cách có ý nghĩa thống kê hay không.
2. **Đánh giá giá trị của biến ngoại sinh (Exogenous Feature Value):**
   - So sánh VAR (nội sinh) vs VARX (nội sinh + ngoại sinh).
   - So sánh LSTM 1 (thuần giá) vs LSTM 2 (giá + thời tiết + vĩ mô).
   - Kiểm định xem việc đưa thêm thông tin sàn London, tỷ giá USD/VND và lượng mưa có cải thiện dự báo có ý nghĩa hay chỉ làm tăng nhiễu.
3. **So sánh mô hình Tuyến tính vs Phi tuyến / Học sâu:**
   - So sánh VARX vs LSTM.
   - So sánh VARX vs TimeXer-DeepVARX.
4. **Đánh giá trên đa hàm mất mát:**
   - Đánh giá trên MSE để xem xét rủi ro biến động mạnh.
   - Đánh giá trên MAE để xem xét mức độ bám sát trung bình thực tế giao dịch của thương lái và nông dân.

---

## 15. Tài liệu tham khảo học thuật (Academic References)

1. **Diebold, F. X., & Mariano, R. S. (1995).** Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263. DOI: `10.1080/07350015.1995.10524599`.
2. **Harvey, D., Leybourne, S., & Newbold, P. (1997).** Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291. DOI: `10.1016/S0169-2070(96)00719-4`.
3. **Newey, W. K., & West, K. D. (1987).** A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3), 703–708. DOI: `10.2307/1913610`.
4. **Newey, W. K., & West, K. D. (1994).** Automatic Lag Selection in Covariance Matrix Estimation. *The Review of Economic Studies*, 61(4), 631–653. DOI: `10.2307/2297912`.
5. **Holm, S. (1979).** A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
6. **Clark, T. E., & West, K. D. (2006).** Using out-of-sample mean squared prediction errors to test the martingale difference hypothesis. *Journal of Econometrics*, 135(1-2), 155–186.
7. **Clark, T. E., & West, K. D. (2007).** Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*, 138(1), 291–311.
8. **Gneiting, T., & Raftery, A. E. (2007).** Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, 102(477), 359–378.
9. **Andrews, D. W. K. (1991).** Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation. *Econometrica*, 59(3), 817–858.
