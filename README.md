# PyTorch-Domain-Transfer-GAN

The re-implementation of diverse domain transfer methods

## 1 Domain Transfer Conclusion

The domain transfer task is slightly different from style transfer task.

There are two main objectives of domain transfer task: multimodality and supervising ability. The multimodality means the model could map an input to diverse domains. While the supervising ability means the model needs paired data of target domain. So there are four categories for general models.

|  | Supervised | Unsupervised |
| :-----:| :-----: | :-----: |
| Unimodal | pix2pix, CRN, task-specific GAN | UNIT, Coupled GAN, DTN, DiscoGAN, DualGAN, CycleGAN, StarGAN |
| Multimodal | pix2pixHD, vid2vid, BicycleGAN | MUNIT, FUNIT |

## 2 About this Repo

- Pix2Pix

- CycleGAN

- StarGAN

- Other models are pendding
