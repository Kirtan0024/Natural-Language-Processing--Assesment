const $ = (id) => document.getElementById(id);
const reviewText = $("reviewText");
const message = $("message");
let lastClassification = null;

reviewText.addEventListener("input", () => {
  $("charCount").textContent = `${reviewText.value.length} / 1200`;
});

function showMessage(text, type = "success") {
  message.textContent = text;
  message.className = `message ${type}`;
}
function hideMessage() { message.className = "message hidden"; }

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: {"Content-Type": "application/json"},
    ...options
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Something went wrong.");
  return data;
}

function renderResult(data) {
  lastClassification = data;
  $("resultCard").classList.remove("hidden");
  $("sentimentBadge").textContent = data.sentiment;
  $("sentimentBadge").className = `badge ${data.sentiment === "Negative" ? "negative" : ""}`;
  $("categoryBadge").textContent = data.category;
  $("processedText").textContent = data.processed_text || "(no content)";
}

async function classify(save = false) {
  const text = reviewText.value.trim();
  hideMessage();
  if (!text) {
    showMessage("Please enter a customer review first.", "error");
    reviewText.focus();
    return;
  }
  const button = save ? $("saveBtn") : $("classifyBtn");
  button.disabled = true;
  const original = button.textContent;
  button.textContent = save ? "Saving..." : "Classifying...";
  try {
    if (save) {
      const data = await api("/api/reviews", {method: "POST", body: JSON.stringify({text})});
      renderResult(data.review);
      showMessage("Review classified and saved successfully.", "success");
      reviewText.value = "";
      $("charCount").textContent = "0 / 1200";
      await refreshAll();
    } else {
      const data = await api("/api/classify", {method: "POST", body: JSON.stringify({text})});
      renderResult(data);
      showMessage("Review classified. Use “Classify & save” to include it in reports.", "success");
    }
  } catch (error) {
    showMessage(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function renderSummary(summary) {
  $("totalReviews").textContent = summary.total_reviews;
  $("positiveCount").textContent = summary.sentiment_counts.Positive || 0;
  $("negativeCount").textContent = summary.sentiment_counts.Negative || 0;
  const categories = summary.category_counts || {};
  const top = Object.entries(categories).sort((a,b) => b[1]-a[1]).find((entry) => entry[1] > 0);
  $("topCategory").textContent = top ? top[0] : "—";
  const styles = {"Delivery":"delivery","Food Quality":"food","App":"app","General":"general"};
  const total = summary.total_reviews || 0;
  $("categoryBreakdown").innerHTML = Object.entries(categories).map(([label, count]) => {
    const pct = total ? Math.round(count / total * 100) : 0;
    return `<div class="breakdown-row"><span class="breakdown-label">${label}</span><div class="track"><div class="fill ${styles[label] || ""}" style="width:${pct}%"></div></div><span class="breakdown-count">${count}</span></div>`;
  }).join("");
  const words = summary.top_words || [];
  $("topWords").innerHTML = words.length
    ? words.map(item => `<span class="word-chip">${escapeHtml(item.word)}<b>${item.count}</b></span>`).join("")
    : `<span class="empty-state">Add reviews to see frequent words.</span>`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[char]));
}
function renderReviews(reviews) {
  const tbody = $("reviewRows");
  if (!reviews.length) {
    tbody.innerHTML = `<tr><td colspan="4" class="empty-table">No reviews yet. Add one above to get started.</td></tr>`;
    return;
  }
  tbody.innerHTML = reviews.map(review => `
    <tr>
      <td>${escapeHtml(review.text)}</td>
      <td><span class="table-badge ${review.sentiment === "Negative" ? "neg" : ""}">${escapeHtml(review.sentiment)}</span></td>
      <td><span class="table-badge table-category">${escapeHtml(review.category)}</span></td>
      <td class="processed-cell">${escapeHtml(review.processed_text)}</td>
    </tr>`).join("");
}
async function refreshAll() {
  try {
    const [summary, history] = await Promise.all([api("/api/summary"), api("/api/reviews")]);
    renderSummary(summary);
    renderReviews(history.reviews);
  } catch (error) {
    showMessage(error.message, "error");
  }
}
$("classifyBtn").addEventListener("click", () => classify(false));
$("saveBtn").addEventListener("click", () => classify(true));
$("refreshBtn").addEventListener("click", refreshAll);
refreshAll();
