import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms

# ==============================================================================
# 수정 가능한 설정값
# ==============================================================================
SAVE_PATH   = 'transformer_blockdiagram.png'
DPI         = 150

# 모든 크기 단위: 인치 (figure 좌표)
FIG_W       = 6.0
BOX_W_MAIN  = 2.8
BOX_W_INNER = 2.5
BOX_H_SM    = 0.48
BOX_H_LG    = 0.58
GAP         = 0.26
ENC_PAD     = 0.20
MARGIN_BOT  = 0.55
MARGIN_TOP  = 0.50

COL_GRAY    = '#E8E8E8'
COL_TEAL    = '#D4EDE8'
COL_PURPLE  = '#ECEAF8'
COL_BLUE    = '#D4E4F4'
COL_CORAL   = '#F4E4D4'
COL_LGRAY   = '#F2F2F2'
COL_BADGE   = '#7A5CC4'

FONT_TITLE  = 9
FONT_SUB    = 7

# ==============================================================================
# y 위치 계산 (아래에서 위)
# ==============================================================================
def top(cy, h): return cy + h/2
def bot(cy, h): return cy - h/2

pos = {}
y = MARGIN_BOT
pos['inp']  = y + BOX_H_SM/2;  y = top(pos['inp'], BOX_H_SM) + GAP
pos['emb']  = y + BOX_H_LG/2;  y = top(pos['emb'], BOX_H_LG) + GAP
pos['pe']   = y + BOX_H_LG/2;  y = top(pos['pe'],  BOX_H_LG) + GAP*1.5   # encoder 진입 여백

enc_y0 = y - ENC_PAD  # encoder 외곽 아래쪽

pos['attn'] = y + BOX_H_LG/2;  y = top(pos['attn'], BOX_H_LG) + GAP
pos['add1'] = y + BOX_H_SM/2;  y = top(pos['add1'], BOX_H_SM) + GAP
pos['ffn']  = y + BOX_H_LG/2;  y = top(pos['ffn'],  BOX_H_LG) + GAP
pos['add2'] = y + BOX_H_SM/2;  y = top(pos['add2'], BOX_H_SM)

enc_y1 = y + ENC_PAD  # encoder 외곽 위쪽

y += GAP*1.5
pos['lin2'] = y + BOX_H_LG/2;  y = top(pos['lin2'], BOX_H_LG) + GAP
pos['out']  = y + BOX_H_SM/2;  y = top(pos['out'],  BOX_H_SM)

title_y = y + 0.38
FIG_H   = title_y + MARGIN_TOP

CX = FIG_W / 2

# ==============================================================================
# figure 생성: axes가 figure 전체를 채우도록
# ==============================================================================
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax  = fig.add_axes([0, 0, 1, 1])   # [left, bottom, width, height] in figure fraction
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis('off')

# ==============================================================================
# 헬퍼
# ==============================================================================
def draw_box(cx, cy, w, h, title, sub=None, fc='#E8E8E8', ec='#888', lw=0.7):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle="round,pad=0,rounding_size=0.05",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3))
    if sub:
        ax.text(cx, cy+h*0.13, title, ha='center', va='center',
                fontsize=FONT_TITLE, fontweight='bold', zorder=4)
        ax.text(cx, cy-h*0.20, sub,   ha='center', va='center',
                fontsize=FONT_SUB,   color='#555', zorder=4)
    else:
        ax.text(cx, cy, title, ha='center', va='center',
                fontsize=FONT_TITLE, fontweight='bold', zorder=4)

def draw_arrow(cx, y0, y1):
    ax.annotate('', xy=(cx, y1), xytext=(cx, y0),
        arrowprops=dict(arrowstyle='->', color='#666', lw=1.0, mutation_scale=8))

def draw_residual(xr, y_top, y_bot):
    kw = dict(color='#bbb', lw=0.8, ls='--', zorder=2)
    ax.plot([xr, xr],      [y_top, y_bot],  **kw)
    ax.plot([xr, xr+0.18], [y_top, y_top],  **kw)
    ax.annotate('', xy=(xr+0.18, y_bot), xytext=(xr, y_bot),
        arrowprops=dict(arrowstyle='->', color='#bbb', lw=0.8, mutation_scale=7))
    ax.text(xr-0.12, (y_top+y_bot)/2, '+',
            ha='center', va='center', fontsize=10, color='#bbb', fontweight='bold')

# ==============================================================================
# 그리기
# ==============================================================================

# --- 인코더 외곽 박스 ---
ax.add_patch(FancyBboxPatch(
    (CX - BOX_W_MAIN/2 - 0.26, enc_y0),
    BOX_W_MAIN + 0.52, enc_y1 - enc_y0,
    boxstyle="round,pad=0,rounding_size=0.10",
    linewidth=1.0, edgecolor='#9070C8',
    facecolor=COL_PURPLE, zorder=1))
ax.text(CX - BOX_W_MAIN/2 - 0.14, (enc_y0+enc_y1)/2,
        'Transformer\nencoder layer',
        ha='center', va='center', fontsize=7,
        color='#5A3A9A', fontweight='bold', rotation=90, zorder=4)

# ×9 배지
bx = CX + BOX_W_MAIN/2 + 0.06
by = (enc_y0+enc_y1)/2
ax.add_patch(FancyBboxPatch((bx, by-0.16), 0.44, 0.32,
    boxstyle="round,pad=0,rounding_size=0.07",
    linewidth=1.0, edgecolor=COL_BADGE, facecolor=COL_BADGE, zorder=5))
ax.text(bx+0.22, by, '× 9',
        ha='center', va='center', fontsize=8.5,
        color='white', fontweight='bold', zorder=6)

# --- 블록들 ---
ax.text(CX, bot(pos['inp'], BOX_H_SM)-0.16,
        '25-dim × 20 timesteps  (TOA, TDOA, DOA)',
        ha='center', va='center', fontsize=6.5, color='#555')
draw_box(CX, pos['inp'], BOX_W_MAIN, BOX_H_SM, 'Input feature sequence', fc=COL_GRAY)
draw_arrow(CX, top(pos['inp'], BOX_H_SM), bot(pos['emb'], BOX_H_LG))

draw_box(CX, pos['emb'], BOX_W_MAIN, BOX_H_LG, 'Linear embedding',
         sub='25  →  128   (scaled by √d_model)', fc=COL_TEAL)
draw_arrow(CX, top(pos['emb'], BOX_H_LG), bot(pos['pe'], BOX_H_LG))

draw_box(CX, pos['pe'], BOX_W_MAIN, BOX_H_LG, 'Positional encoding',
         sub='Sinusoidal,  max_len = 20', fc=COL_TEAL)
draw_arrow(CX, top(pos['pe'], BOX_H_LG), bot(pos['attn'], BOX_H_LG))

draw_box(CX, pos['attn'], BOX_W_INNER, BOX_H_LG, 'Multi-head self-attention',
         sub='N_H = 8,  d_k = d_v = 16,  d_model = 128', fc=COL_BLUE)
draw_arrow(CX, top(pos['attn'], BOX_H_LG), bot(pos['add1'], BOX_H_SM))

xr = CX - BOX_W_INNER/2 - 0.33
draw_residual(xr, top(pos['pe'], BOX_H_LG)+0.04, bot(pos['add1'], BOX_H_SM))

draw_box(CX, pos['add1'], BOX_W_INNER, BOX_H_SM, 'Add & layer norm', fc=COL_LGRAY)
draw_arrow(CX, top(pos['add1'], BOX_H_SM), bot(pos['ffn'], BOX_H_LG))

draw_box(CX, pos['ffn'], BOX_W_INNER, BOX_H_LG, 'Feed-forward network',
         sub='d_ff = 512,  GELU,  dropout = 0.0534', fc=COL_CORAL)
draw_arrow(CX, top(pos['ffn'], BOX_H_LG), bot(pos['add2'], BOX_H_SM))

draw_residual(xr, top(pos['add1'], BOX_H_SM)+0.04, bot(pos['add2'], BOX_H_SM))

draw_box(CX, pos['add2'], BOX_W_INNER, BOX_H_SM, 'Add & layer norm', fc=COL_LGRAY)
draw_arrow(CX, enc_y1, bot(pos['lin2'], BOX_H_LG))

draw_box(CX, pos['lin2'], BOX_W_MAIN, BOX_H_LG, 'Linear output',
         sub='128  →  3', fc=COL_TEAL)
draw_arrow(CX, top(pos['lin2'], BOX_H_LG), bot(pos['out'], BOX_H_SM))

draw_box(CX, pos['out'], BOX_W_MAIN, BOX_H_SM,
         'Output:  (x, y, z)  ×  20 steps', fc=COL_GRAY)

ax.text(CX, title_y, 'Transformer Encoder Architecture',
        ha='center', va='center', fontsize=11, fontweight='bold')

# ==============================================================================
plt.savefig(SAVE_PATH, dpi=DPI, facecolor='white', edgecolor='none')
plt.close()
print(f"저장 완료: {SAVE_PATH}  ({FIG_W:.1f} × {FIG_H:.2f} inch)")