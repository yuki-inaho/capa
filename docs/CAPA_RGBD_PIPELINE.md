# CAPA RGB-D refinement pipeline

この文書は、RGB画像、センサ深度、CAPAのMoGe-2 refinement結果を、別々の点群として保存する処理の前提を記録する。

## 入力データと役割

| 入力 | 想定形式 | 役割 |
|---|---|---|
| RGB画像 | RGB PNG、aligned depthと同じ解像度 | PLYの頂点色と画素座標を提供 |
| raw aligned depth | RGB視野へ整列済み、dilation前、単一チャンネルuint16 PNG | 生センサ観測点群の深度。正値画素だけを観測点とする |
| CAPA depth | `depth_pred_nhw`を持つ`.pt`、単位m | MoGe-2 VPTで補完・適応したdense depth |
| RGB camera YAML | `K`を含むカメラパラメータ | RGBカメラ座標系への逆投影 |

センサ深度のuint16値は、既定では`depth_scale=1000`としてメートルへ変換する。つまり、画素値750は0.750 mとして扱う。実センサ仕様と異なる場合は`--depth-scale`を変更する。

## 処理内容

1. RGBとaligned depthの画像サイズが一致することを検証する。
2. raw aligned depthをuint16からfloat32のメートルへ変換する。
3. CAPAの予測テンソルを読み、正値かつ有限の画素を有効とする。
4. RGBカメラの内部パラメータ`fx, fy, cx, cy`で各画素を逆投影する。

   ```text
   z = depth[v, u]
   x = (u - cx) * z / fx
   y = (v - cy) * z / fy
   P = (x, y, z)
   ```

5. 生観測点群PLYとCAPA refined点群PLYを別ファイルへ保存する。

この処理ではfusion、観測値の上書き、許容誤差によるclampは行わない。raw PLYはセンサ観測の記録、refined PLYはCAPA/MoGe-2の出力として比較可能にする。

## 座標系と出力

PLYの座標系はOpenCVカメラ座標系（x右、y下、z前）とする。両PLYともRGB画像の色を頂点色として格納する。raw PLYは疎、CAPA refined PLYは通常denseになる。

再現コマンド例:

```bash
venv/moge-2_env/bin/python scripts/export_rgbd_capa_ply.py \
  --rgb <rgb.png> \
  --raw-depth <aligned_depth_before_dilation.png> \
  --refined-depth <capa_prediction.pt> \
  --camera <rgb_camera_param.yaml> \
  --raw-out <raw_observation.ply> \
  --refined-out <capa_moge_refined.ply>
```
