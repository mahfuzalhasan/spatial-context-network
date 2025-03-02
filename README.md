### [Automated Segmentation of Lymph Nodes on Neck CT Scans Using Deep Learning](https://link.springer.com/article/10.1007/s10278-024-01114-w)
Official Pytorch implementation of S-Net, from the following paper:
[Automated Segmentation of Lymph Nodes on Neck CT Scans Using Deep Learning](https://link.springer.com/article/10.1007/s10278-024-01114-w). Journal of Imaging Informatics in Medicine 2023 (Accepted)

---

<p align="center">
<img src="figures/figure_1.png" width=100% height=40% 
class="center">
</p>

<p align="center">
<img src="figures/figure_2.jpg" width=100% height=40% 
class="center">
</p>

We propose **Spatial Context Network (S-Net)**, a dilated convolution-based network to capture multi-scale context from head and neck CT for small lymph node (LN) segmentation. The network downsamples feature space only twice in encoding stage to preserve the spatial context from small LN. On the other hand, to address the limited receptive field, the network utilizes Atrous Spatial Pyramid Pooling (ASPP) to focus on multi-scale feature. As skip connection, the network utilizes reverse axial attention module to filter out unnecessary component detection outside of LN in order to reduce False Positive (FP). The network is supervised by binary tversky loss and weighted iou (termed as structure loss) [PraNet](https://arxiv.org/abs/2006.11392) to address the per-sample class-imbalance problem.
<!-- 

    ```
    git clone https://github.com/mahfuzalhasan/spatial-context-network.git
    ```
2. Run *python3 main.py* to check the model loading.
3. Torch version 1.12.1 is used with CUDA 10.2. But latest version should work just fine.  -->