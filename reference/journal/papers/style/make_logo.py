"""Draw the Shisso wordmark (mark + 'Shisso Robotics' + small grey 質素) to logo/shisso_wordmark.{pdf,svg}. Matches site/public/assets/wordmark.svg."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.font_manager import FontProperties, findfont
from pathlib import Path
ACCENT, INK, GREY = "#1F5FBF", "#1A1C20", "#8A8F99"
out = Path(__file__).parent / "logo"
fig = plt.figure(figsize=(3.2, 0.42))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 320); ax.set_ylim(0, 42); ax.axis("off")
ax.add_patch(FancyBboxPatch((2, 6), 30, 30, boxstyle="round,pad=0,rounding_size=7", fc=ACCENT, ec="none"))
ax.add_patch(Circle((17, 21), 6.5, fc="none", ec="white", lw=3.2))
ax.text(42, 21, "Shisso", fontsize=14, fontweight="bold", color=INK, va="center")
ax.text(118, 21, "Robotics", fontsize=14, color=INK, va="center")
cjk = next((f for f in ("Noto Sans CJK JP", "AR PL UMing CN", "AR PL UMing TW", "IPAGothic", "TakaoGothic") if "DejaVu" not in findfont(FontProperties(family=f), fallback_to_default=True)), None)
ax.text(208, 20, "質素", fontsize=9, color=GREY, va="center", **({"family": cjk} if cjk else {}))
fig.savefig(out / "shisso_wordmark.pdf"); fig.savefig(out / "shisso_wordmark.svg")
print("cjk font:", cjk)
