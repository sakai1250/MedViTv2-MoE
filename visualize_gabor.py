import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import seaborn as sns
from MedViTv3 import MedViTv3_tiny, MedViTv3_small, MedViTv3_base, MedViTv3_large
from ac_kan import ComplexGaborConv2d

model_classes = {
    'MedViTv3_tiny': MedViTv3_tiny,
    'MedViTv3_small': MedViTv3_small,
    'MedViTv3_base': MedViTv3_base,
    'MedViTv3_large': MedViTv3_large
}

# パラメータごとの日本語説明と単位
PARAM_INFO = {
    'sigma': {
        'title': 'σ (スケール / ガウス包絡線の幅)',
        'xlabel': 'σ 値',
        'description': '値が大きい → 広い受容野（低周波寄り）\n値が小さい → 狭い受容野（高周波寄り）',
    },
    'omega': {
        'title': 'ω (キャリア周波数)',
        'xlabel': 'ω 値 [rad]',
        'description': '値が大きい → 高周波成分（細かいテクスチャ）の検出\n値が小さい → 低周波成分（なだらかな変化）の検出',
    },
    'theta': {
        'title': 'θ (方向 / フィルタの向き)',
        'xlabel': 'θ 値 [rad]',
        'description': '0 → 水平エッジ, π/2 → 垂直エッジ\n学習後に特定方向に偏っていると、その方向のエッジに特化',
    },
    'psi': {
        'title': 'ψ (位相シフト)',
        'xlabel': 'ψ 値 [rad]',
        'description': '実部と虚部の位相差を決定\n偶数・奇数フィルタのバランスを制御',
    },
}

def make_short_label(layer_name):
    """長いレイヤー名を短い表示名に変換"""
    # e.g. "features.0.ffn.net.3.layer" -> "Stage0/Block0"
    parts = layer_name.split('.')
    # features.{idx} からstage/blockを推定
    try:
        feat_idx = int(parts[1])
        # MedViTv3_tiny depths=[2,2,6,1] -> cumsum=[2,4,10,11]
        cumsum = [2, 4, 10, 11]
        stage = 0
        for i, c in enumerate(cumsum):
            if feat_idx < c:
                stage = i
                block = feat_idx - (cumsum[i-1] if i > 0 else 0)
                break
        return f"S{stage}/B{block}"
    except (IndexError, ValueError):
        # フォールバック: 短縮名
        return layer_name.split('.')[-2] if len(parts) > 2 else layer_name


def plot_histograms(params_dict, save_dir):
    """各パラメータの分布をヒストグラム + KDEで描画（凡例・説明付き）"""
    os.makedirs(save_dir, exist_ok=True)
    
    for param_name, info in PARAM_INFO.items():
        fig, ax = plt.subplots(figsize=(14, 7))
        
        for layer_name, params in params_dict.items():
            vals = params[param_name]
            short_name = make_short_label(layer_name)
            sns.kdeplot(vals, label=short_name, fill=True, alpha=0.15, linewidth=1.5, ax=ax)
        
        ax.set_title(info['title'], fontsize=16, fontweight='bold')
        ax.set_xlabel(info['xlabel'], fontsize=13)
        ax.set_ylabel('密度 (Density)', fontsize=13)
        
        # 凡例を右上に配置
        ax.legend(title='レイヤー', fontsize=9, title_fontsize=10,
                  loc='upper right', framealpha=0.9)
        
        # 説明テキストを図の下部に追加
        fig.text(0.02, -0.02, info['description'], fontsize=9,
                 style='italic', color='gray', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'hist_{param_name}_all_layers.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_kernels(layer_name, module, save_dir, num_kernels=24):
    """2Dカーネル（実部・虚部）を可視化（カラーバー・説明付き）"""
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        real, imag = module.get_kernel()
    
    out_ch = real.size(0)
    plot_count = min(out_ch, num_kernels)
    
    cols = int(np.ceil(np.sqrt(plot_count)))
    rows = int(np.ceil(plot_count / cols))
    short_name = make_short_label(layer_name)

    for part_name, data_tensor in [('Real (実部)', real), ('Imag (虚部)', imag)]:
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2 + 1.5))
        fig.suptitle(f'Gabor Kernel {part_name}\nLayer: {short_name} ({layer_name})\n'
                     f'Channels: {out_ch}, Kernel Size: {data_tensor.shape[-1]}x{data_tensor.shape[-1]}',
                     fontsize=13, fontweight='bold', y=1.02)
        
        axes_flat = np.array(axes).flatten() if plot_count > 1 else [axes]
        
        # 全カーネルで共通のカラースケールを使用
        all_vals = data_tensor[:plot_count, 0].detach().cpu().numpy()
        global_max = np.max(np.abs(all_vals)) + 1e-6
        
        for i, ax in enumerate(axes_flat):
            if i < plot_count:
                kernel_img = data_tensor[i, 0].detach().cpu().numpy()
                im = ax.imshow(kernel_img, cmap='coolwarm', vmin=-global_max, vmax=global_max)
                ax.set_title(f'Ch{i}', fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # カラーバーの追加
        cbar = fig.colorbar(im, ax=axes_flat.tolist() if hasattr(axes_flat, 'tolist') else list(axes_flat),
                           shrink=0.8, aspect=30, pad=0.02)
        cbar.set_label('フィルタ応答の強度', fontsize=10)
        
        # 説明テキスト
        desc = ('赤 = 正の応答（明るいエッジ検出）, 青 = 負の応答（暗いエッジ検出）\n'
                '実部: cos成分 (偶数対称フィルタ), 虚部: sin成分 (奇数対称フィルタ)')
        fig.text(0.02, -0.02, desc, fontsize=8, style='italic', color='gray', va='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
        
        suffix = 'real' if 'Real' in part_name else 'imag'
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'kernel_{suffix}_{layer_name.replace(".", "_")}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


def plot_summary_table(params_dict, save_dir):
    """全レイヤーのパラメータ統計を一覧テーブルとして画像保存"""
    os.makedirs(save_dir, exist_ok=True)
    rows_data = []
    row_labels = []
    
    for layer_name, params in params_dict.items():
        short_name = make_short_label(layer_name)
        row_labels.append(short_name)
        row = []
        for p in ['sigma', 'omega', 'theta', 'psi']:
            vals = params[p]
            row.extend([f'{np.mean(vals):.3f}', f'{np.std(vals):.3f}'])
        rows_data.append(row)
    
    col_labels = ['σ mean', 'σ std', 'ω mean', 'ω std', 'θ mean', 'θ std', 'ψ mean', 'ψ std']
    
    fig, ax = plt.subplots(figsize=(14, max(3, len(row_labels) * 0.6 + 2)))
    ax.axis('off')
    ax.set_title('Gaborパラメータ統計サマリー (全レイヤー)', fontsize=14, fontweight='bold', pad=20)
    
    table = ax.table(cellText=rows_data, rowLabels=row_labels, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    
    # ヘッダー色
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    
    # 行ラベル色
    for i in range(len(row_labels)):
        table[i+1, -1].set_facecolor('#D6E4F0')
    
    desc = ('σ: ガウス包絡線の幅, ω: キャリア周波数 [rad], '
            'θ: フィルタ方向 [rad], ψ: 位相シフト [rad]')
    fig.text(0.5, 0.02, desc, fontsize=9, ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'param_summary_table.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Gabor-KAN (ACKAN) パラメータ可視化ツール')
    parser.add_argument('--model_name', type=str, default='MedViTv3_tiny')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='学習済み .pth ファイルのパス')
    parser.add_argument('--num_classes', type=int, default=2, help='分類クラス数')
    parser.add_argument('--save_dir', type=str, default='gabor_visualizations', help='画像の出力先ディレクトリ')
    parser.add_argument('--stages', type=int, nargs='+', default=[0, 1, 2, 3], help='ACKANを適用したStage')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model_name not in model_classes:
        raise ValueError(f"Model {args.model_name} not supported.")
        
    model_class = model_classes[args.model_name]
    net = model_class(num_classes=args.num_classes, use_ackan=True, ackan_stages=args.stages).to(device)

    print(f"Loading weights from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    if 'model' in checkpoint:
        net.load_state_dict(checkpoint['model'], strict=False)
    else:
        net.load_state_dict(checkpoint, strict=False)
    
    net.eval()
    
    params_dict = {}
    gabor_modules = {}

    for name, module in net.named_modules():
        if isinstance(module, ComplexGaborConv2d):
            gabor_modules[name] = module
            params_dict[name] = {
                'sigma': module.sigma.detach().cpu().numpy(),
                'omega': module.omega.detach().cpu().numpy(),
                'theta': module.theta.detach().cpu().numpy(),
                'psi': module.psi.detach().cpu().numpy(),
            }

    print(f"Found {len(gabor_modules)} ComplexGaborConv2d layers.")

    if len(gabor_modules) == 0:
        print("No ComplexGaborConv2d layers found. Did you use the correct model_name and checkpoint?")
        return

    # 1. パラメータ分布
    print(f"[1/3] パラメータ分布ヒストグラムを生成中...")
    plot_histograms(params_dict, args.save_dir)

    # 2. 2Dカーネル
    print(f"[2/3] 2Dカーネル画像を生成中...")
    for name, module in gabor_modules.items():
        plot_kernels(name, module, args.save_dir)

    # 3. サマリーテーブル
    print(f"[3/3] パラメータ統計サマリーテーブルを生成中...")
    plot_summary_table(params_dict, args.save_dir)
            
    print(f"\n=== 可視化完了 ===")
    print(f"出力先: {os.path.abspath(args.save_dir)}/")
    print(f"  - hist_*.png       : 各パラメータの分布（凡例付き）")
    print(f"  - kernel_real_*.png: 実部カーネル（カラーバー付き）")
    print(f"  - kernel_imag_*.png: 虚部カーネル（カラーバー付き）")
    print(f"  - param_summary_table.png : パラメータ統計テーブル")

if __name__ == '__main__':
    main()
