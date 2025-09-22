import argparse
import pathlib
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import os
import tempfile

def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract ESM2 features from sequences in a pickle file and save to a new pickle file"
    )
    parser.add_argument(
        "model_location",
        type=str,
        help="PyTorch model file OR name of pretrained model to download (e.g., esm2_t33_650M_UR50D)",
    )
    parser.add_argument(
        "input_pkl",
        type=pathlib.Path,
        help="Input pickle file containing a 'sequence' column",
    )
    parser.add_argument(
        "output_pkl",
        type=pathlib.Path,
        help="Output pickle file with added 'esm2_features' column",
    )
    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--sequence_col", 
        type=str, 
        default="sequence", 
        help="Name of the sequence column in pkl file"
    )
    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser

def main(args):
    # 加载ESM2模型
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script does not handle MSA Transformer models")
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.nogpu else "cpu")
    model = model.to(device)
    print(f"Model transferred to {device}")
    
    # 读取pkl文件
    print(f"Reading input pickle file: {args.input_pkl}")
    df = pd.read_pickle(args.input_pkl)
    
    if args.sequence_col not in df.columns:
        raise ValueError(f"Column '{args.sequence_col}' not found in input pkl")
    
    # 准备序列数据
    sequences = df[args.sequence_col].tolist()
#    print(sequences )
    sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
    
    # 创建临时FASTA文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
        for seq_id, seq in zip(sequence_ids, sequences):
            f.write(f">{seq_id}\n{seq}\n")
        temp_fasta = f.name
    
    # 加载FASTA数据
    dataset = FastaBatchedDataset.from_file(temp_fasta)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    
    print(f"Read {len(dataset)} sequences from pkl file")
    
    # 层索引处理
    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]
    
    # 特征提取
    all_features = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing batch {batch_idx+1}/{len(batches)}")
            toks = toks.to(device=device, non_blocking=True)
            
            out = model(toks, repr_layers=repr_layers)
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
#            print(representations)
#            print(repr_layers)
            for i, label in enumerate(labels):
                seq_idx = int(label.split("_")[1])
                seq_len = len(strs[i])
                
                # 提取平均特征（使用最后一层）
                if 33 in repr_layers:
                    layer_idx = repr_layers.index(-1) if -1 in repr_layers else 0
                    feature = representations[repr_layers[layer_idx]][i, 1:seq_len+1].clone()
                    # print(feature)
                    # feature = feature.numpy()
                    # print(feature.shape)
                    # feature2 = representations[repr_layers[layer_idx]][i, 1:seq_len + 1].clone()
                    # feature2 = feature2.numpy()
                    # print(feature2)
                    # print(feature2.shape)
#                    feature = feature.to(torch.float16)

                    feature = feature.numpy()
#                    print(feature)
#                    print(1)
#                    print(feature)
#                    print(2)
                    all_features.append(feature)
    
    # 清理临时文件
    os.unlink(temp_fasta)
    
    # 添加特征到DataFrame
    df["esm2_features"] = all_features
    print(f"Extracted features for {len(df)} sequences")
    
    # 保存结果
    output_dir = args.output_pkl.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_pickle(args.output_pkl)
    print(f"Features saved to {args.output_pkl}")

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)