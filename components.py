# components.py - reusable UI components for GlanhzeeTrade.ai

import streamlit as st


def render_metric_card(title, value, delta=None, delta_class="delta-flat", icon=""):
    html = (
        f'<div class="metric-card">'
        f'<div class="metric-title">{icon} {title}</div>'
        f'<div class="metric-value">{value}</div>'
    )
    if delta:
        html += f'<div class="metric-delta {delta_class}">{delta}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_header(ticker_label, is_online=True):
    status_text = "LIVE" if is_online else "OFFLINE"
    status_color = "#10B981" if is_online else "#EF4444"
    status_bg = "rgba(16,185,129,0.1)" if is_online else "rgba(239,68,68,0.1)"
    status_border = "rgba(16,185,129,0.2)" if is_online else "rgba(239,68,68,0.2)"

    html = f"""<div style='background:linear-gradient(145deg,rgba(15,23,42,0.6) 0%,rgba(7,9,14,0.8) 100%);border:1px solid rgba(255,255,255,0.05);border-radius:20px;padding:28px 36px;margin-bottom:28px;display:flex;align-items:center;justify-content:space-between;backdrop-filter:blur(20px);box-shadow:0 8px 40px rgba(0,0,0,0.4),inset 0 1px 0 rgba(255,255,255,0.04);position:relative;overflow:hidden;'>
<div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.2),transparent);"></div>
<div>
<div style='display:inline-flex;align-items:center;gap:8px;padding:5px 14px;background:rgba(56,189,248,0.08);border:1px solid rgba(56,189,248,0.18);border-radius:50px;font-size:10px;font-weight:700;color:#38BDF8;margin-bottom:14px;letter-spacing:2.5px;text-transform:uppercase;'>
<span style='width:5px;height:5px;border-radius:50%;background:#38BDF8;box-shadow:0 0 8px #38BDF8;' class="pulse-dot"></span>
{ticker_label} · Intelligence Terminal
</div>
<h1 style='font-size:28px;font-weight:800;color:#F8FAFC;margin:0;letter-spacing:-0.5px;font-family:"Outfit",sans-serif;line-height:1.2;'>
Real-Time Market <span style='background:linear-gradient(135deg,#38BDF8,#818CF8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Intelligence</span>
</h1>
</div>
<div style='display:flex;flex-direction:column;align-items:flex-end;gap:8px;'>
<div style='font-family:"JetBrains Mono",monospace;font-size:10px;color:#334155;letter-spacing:2px;text-transform:uppercase;'>System Status</div>
<div style='display:flex;align-items:center;gap:8px;background:{status_bg};padding:8px 18px;border-radius:10px;border:1px solid {status_border};color:{status_color};font-weight:700;font-size:12px;letter-spacing:2px;'>
<div style='width:8px;height:8px;border-radius:50%;background:{status_color};box-shadow:0 0 10px {status_color};' class="pulse-dot"></div>
{status_text}
</div>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_ai_insight(insight_text, confidence):
    if confidence > 0.6:
        conf_color, conf_bg = "#10B981", "rgba(16,185,129,0.08)"
    elif confidence > 0.4:
        conf_color, conf_bg = "#F59E0B", "rgba(245,158,11,0.08)"
    else:
        conf_color, conf_bg = "#EF4444", "rgba(239,68,68,0.08)"

    bars = int(confidence * 10)
    bar_html = "".join([
        f'<div style="width:6px;height:{12+i*2}px;border-radius:3px;background:{"#10B981" if i < bars else "rgba(255,255,255,0.08)"};transition:all 0.3s;"></div>'
        for i in range(10)
    ])

    html = f"""<div style='padding:24px 28px;background:linear-gradient(145deg,rgba(15,23,42,0.7),rgba(7,9,14,0.9));border:1px solid rgba(56,189,248,0.15);border-radius:18px;margin-bottom:20px;box-shadow:0 8px 30px rgba(0,0,0,0.3),inset 0 1px 0 rgba(56,189,248,0.05);position:relative;overflow:hidden;'>
<div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(56,189,248,0.3),transparent);"></div>
<div style='display:flex;align-items:center;gap:12px;margin-bottom:16px;'>
<div style='width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,rgba(56,189,248,0.15),rgba(129,140,248,0.15));border:1px solid rgba(56,189,248,0.15);display:flex;align-items:center;justify-content:center;font-size:20px;'>🤖</div>
<div>
<div style='font-size:15px;font-weight:700;color:#F8FAFC;'>AI Market Analysis</div>
<div style='font-size:11px;color:#475569;letter-spacing:1px;text-transform:uppercase;'>Powered by Deep Learning</div>
</div>
</div>
<p style='margin:0 0 20px;color:#94A3B8;font-size:14px;line-height:1.7;font-weight:300;'>{insight_text}</p>
<div style='display:flex;align-items:center;gap:14px;background:{conf_bg};padding:10px 16px;border-radius:10px;border:1px solid rgba(255,255,255,0.05);'>
<div style='display:flex;align-items:flex-end;gap:2px;height:24px;'>{bar_html}</div>
<span style='font-size:12px;color:#64748B;letter-spacing:1px;text-transform:uppercase;'>Confidence</span>
<span style='font-size:15px;font-weight:800;color:{conf_color};font-family:"JetBrains Mono",monospace;'>{confidence:.0%}</span>
</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_news_card(headline, date, source, score):
    if score > 0.1:
        color, icon, label = "#10B981", "🟢", "Bullish"
    elif score < -0.1:
        color, icon, label = "#EF4444", "🔴", "Bearish"
    else:
        color, icon, label = "#64748B", "⚪", "Neutral"

    html = f"""<div class="news-card" style='padding:16px 18px;background:rgba(15,23,42,0.5);border:1px solid rgba(255,255,255,0.04);border-left:3px solid {color};border-radius:12px;margin-bottom:10px;transition:all 0.3s ease;cursor:pointer;box-shadow:-3px 0 15px rgba(0,0,0,0.1);'>
<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
<span style='font-size:11px;color:#334155;font-weight:600;letter-spacing:0.5px;'>{source} · {date}</span>
<span style='font-size:11px;font-weight:700;color:{color};background:rgba(0,0,0,0.2);padding:2px 8px;border-radius:6px;border:1px solid rgba(255,255,255,0.04);'>{icon} {label} {score:+.2f}</span>
</div>
<div style='font-size:14px;color:#CBD5E1;font-weight:400;line-height:1.5;'>{headline}</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_risk_gauge(risk_score, label):
    if risk_score < 40:
        color, glow = "#10B981", "rgba(16,185,129,0.2)"
    elif risk_score < 70:
        color, glow = "#F59E0B", "rgba(245,158,11,0.2)"
    else:
        color, glow = "#EF4444", "rgba(239,68,68,0.2)"

    r = 54
    circ = 2 * 3.14159 * r
    dash = circ * risk_score / 100
    html = f"""<div class="metric-card" style="text-align:center;">
<div class="metric-title" style="justify-content:center;">⚡ RISK SCORE</div>
<svg width="130" height="80" viewBox="0 0 130 80" style="margin:0 auto;display:block;">
<path d="M 15,75 A 54,54 0 0,1 115,75" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="8" stroke-linecap="round"/>
<path d="M 15,75 A 54,54 0 0,1 115,75" fill="none" stroke="{color}" stroke-width="8" stroke-linecap="round" stroke-dasharray="{dash * 1.57:.1f} 999" style="filter:drop-shadow(0 0 6px {color});transition:stroke-dasharray 1s ease;"/>
<text x="65" y="68" text-anchor="middle" font-size="22" font-weight="800" font-family="JetBrains Mono" fill="{color}">{risk_score}</text>
</svg>
<div style="font-size:12px;color:{color};font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-top:4px;">{label}</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)


def render_section_header(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def render_landing():
    """Full animated landing page with particles, counters, and typewriter effect."""
    html = """<canvas id="particles-canvas"></canvas>
<div class="landing-wrapper">
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>
<div class="grid-overlay"></div>
<div class="landing-content">
<div class="landing-badge">
<div class="badge-dot"></div>
Live Intelligence Platform
</div>
<div class="landing-title">GlanhzeeTrade.ai</div>
<div class="landing-title-line2" id="typewriter-target">Market Intelligence</div>
<p class="landing-subtitle">
Institutional-grade <b>NLP sentiment analysis</b>, deep learning forecasts,
and real-time technical charting — built for serious traders.
</p>
<div class="stats-row">
<div class="stat-item">
<div class="stat-number" data-target="50">50+</div>
<div class="stat-label">News Sources Daily</div>
</div>
<div class="stat-item">
<div class="stat-number" data-target="3">3</div>
<div class="stat-label">ML Models</div>
</div>
<div class="stat-item">
<div class="stat-number" data-target="5">5</div>
<div class="stat-label">Day Forecast</div>
</div>
<div class="stat-item">
<div class="stat-number" data-target="99">99.9%</div>
<div class="stat-label">% Uptime</div>
</div>
</div>
<div class="features-grid">
<div class="feature-box">
<div class="feature-icon-wrapper">🌍</div>
<div class="feature-title">Global Sentiment Engine</div>
<div class="feature-desc">Real-time NLP across Yahoo Finance, Economic Times &amp; Moneycontrol. VADER + TextBlob dual-scoring with finance-tuned lexicon.</div>
</div>
<div class="feature-box">
<div class="feature-icon-wrapper">🧠</div>
<div class="feature-title">LSTM Deep Learning</div>
<div class="feature-desc">Stacked LSTM neural networks trained on sequential price-action data for short-term directional prediction with confidence scoring.</div>
</div>
<div class="feature-box">
<div class="feature-icon-wrapper">⚡</div>
<div class="feature-title">Live Market Screener</div>
<div class="feature-desc">Instant top-gainer and loser detection across Tech, Crypto, Energy, Finance &amp; Automotive sectors in a single view.</div>
</div>
</div>
</div>
</div>
<script>
(function() {
var canvas = document.getElementById('particles-canvas');
if (!canvas) return;
var ctx = canvas.getContext('2d');
var W, H, particles = [];
function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
resize();
window.addEventListener('resize', resize);
function Particle() { this.reset(); }
Particle.prototype.reset = function() {
this.x = Math.random() * W; this.y = Math.random() * H;
this.r = Math.random() * 1.5 + 0.3; this.speed = Math.random() * 0.3 + 0.05;
this.angle = Math.random() * Math.PI * 2; this.opacity = Math.random() * 0.4 + 0.05;
this.color = Math.random() > 0.5 ? '56,189,248' : '129,140,248';
};
Particle.prototype.update = function() {
this.y -= this.speed;
this.x += Math.sin(this.angle + Date.now()*0.0005) * 0.3;
if (this.y < -5) { this.reset(); this.y = H + 5; }
};
for (var i = 0; i < 80; i++) { particles.push(new Particle()); }
function drawLines() {
for (var a = 0; a < particles.length; a++) {
for (var b = a+1; b < particles.length; b++) {
var dx = particles[a].x - particles[b].x;
var dy = particles[a].y - particles[b].y;
var d = Math.sqrt(dx*dx + dy*dy);
if (d < 120) {
ctx.beginPath();
ctx.strokeStyle = 'rgba(56,189,248,' + (0.06*(1-d/120)) + ')';
ctx.lineWidth = 0.5;
ctx.moveTo(particles[a].x, particles[a].y);
ctx.lineTo(particles[b].x, particles[b].y);
ctx.stroke();
}}}}
function animate() {
ctx.clearRect(0,0,W,H);
particles.forEach(function(p) {
p.update();
ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
ctx.fillStyle='rgba('+p.color+','+p.opacity+')'; ctx.fill();
});
drawLines(); requestAnimationFrame(animate);
}
animate();
function animateCounter(el, target, duration) {
var start = 0; var step = target / (duration / 16);
var timer = setInterval(function() {
start += step;
if (start >= target) { start = target; clearInterval(timer); }
el.textContent = Math.floor(start);
}, 16);
}
setTimeout(function() {
document.querySelectorAll('.stat-number[data-target]').forEach(function(el) {
animateCounter(el, parseInt(el.dataset.target), 1800);
});
}, 600);
var phrases = ['Market Intelligence','Sentiment Analysis','LSTM Forecasting','Risk Analytics'];
var ti = 0, ci = 0, deleting = false;
var tw = document.getElementById('typewriter-target');
function typeWrite() {
if (!tw) return;
var phrase = phrases[ti % phrases.length];
if (!deleting) {
tw.textContent = phrase.slice(0, ++ci);
if (ci >= phrase.length) { deleting = true; setTimeout(typeWrite, 2000); return; }
} else {
tw.textContent = phrase.slice(0, --ci);
if (ci <= 0) { deleting = false; ti++; setTimeout(typeWrite, 400); return; }
}
setTimeout(typeWrite, deleting ? 50 : 80);
}
setTimeout(typeWrite, 1200);
})();
</script>"""
    st.markdown(html, unsafe_allow_html=True)


def render_prediction_signal(ticker_lbl, pred, prob, best_model):
    pill_text = "BULLISH" if pred == 1 else "BEARISH"

    if pred == 1:
        conf_color = "#10B981"
        shadow = "0 0 80px rgba(16,185,129,0.2)"
        bg = "linear-gradient(135deg,rgba(16,185,129,0.08),rgba(0,0,0,0))"
        border = "rgba(16,185,129,0.2)"
        anim = "glow-green"
        arrow = "↑"
    else:
        conf_color = "#EF4444"
        shadow = "0 0 80px rgba(239,68,68,0.2)"
        bg = "linear-gradient(135deg,rgba(239,68,68,0.08),rgba(0,0,0,0))"
        border = "rgba(239,68,68,0.2)"
        anim = "glow-red"
        arrow = "↓"

    bars = "".join([
        f'<div style="flex:1;height:{int(5+prob*20*(i/10)):.0f}px;background:{"rgba(255,255,255,0.6)" if i/10 < prob else "rgba(255,255,255,0.06)"};border-radius:3px;transition:height 0.5s ease {i*0.05:.1f}s;"></div>'
        for i in range(1, 11)
    ])

    html = f"""<div style="text-align:center;padding:48px 36px;border-radius:24px;background:{bg};border:1px solid {border};box-shadow:{shadow};animation:{anim} 3s ease-in-out infinite;margin-top:20px;position:relative;overflow:hidden;">
<div style="position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,{conf_color}40,transparent);"></div>
<div style="font-size:11px;color:#334155;letter-spacing:3px;text-transform:uppercase;margin-bottom:16px;">Deep Learning Signal — {ticker_lbl}</div>
<div style="font-size:72px;line-height:1;margin-bottom:4px;">{arrow}</div>
<div style="font-size:60px;font-weight:900;color:{conf_color};letter-spacing:4px;text-shadow:0 0 40px {conf_color}80;font-family:'JetBrains Mono',monospace;margin-bottom:12px;">{pill_text}</div>
<p style="color:#94A3B8;font-size:16px;margin:0 0 24px;font-weight:300;">Confidence: <b style="color:{conf_color};font-weight:800;font-size:20px;font-family:'JetBrains Mono',monospace;">{prob:.1%}</b></p>
<div style="display:flex;align-items:flex-end;gap:4px;height:30px;max-width:200px;margin:0 auto 20px;">{bars}</div>
<div style="font-size:11px;color:#334155;letter-spacing:2px;">ACTIVE ENGINE: <b style="color:#A5B4FC;">{best_model.upper()}</b></div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)
