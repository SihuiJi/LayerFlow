DeepSpeedEngine(
  (module): SATVideoDiffusionEngine(
    (model): OpenAIWrapper(
      (diffusion_model): DiffusionTransformer(
        (mixins): ModuleDict(
          (pos_embed): Basic3DPositionEmbeddingMixin()
          (patch_embed): ImagePatchEmbeddingMixin_layer(
            (proj): Conv2d(16, 1920, kernel_size=(2, 2), stride=(2, 2))
            (text_proj): Linear(in_features=4096, out_features=1920, bias=True)
          )
          (adaln_layer): AdaLNMixin(
            (adaLN_modulations): ModuleList(
              (0-29): 30 x Sequential(
                (0): SiLU()
                (1): Linear(in_features=512, out_features=23040, bias=True)
              )
            )
            (query_layernorm_list): ModuleList(
              (0-29): 30 x LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            )
            (key_layernorm_list): ModuleList(
              (0-29): 30 x LayerNorm((64,), eps=1e-06, elementwise_affine=True)
            )
          )
          (final_layer): FinalLayerMixin(
            (norm_final): LayerNorm((1920,), eps=1e-06, elementwise_affine=True)
            (linear): Linear(in_features=1920, out_features=64, bias=True)
            (adaLN_modulation): Sequential(
              (0): SiLU()
              (1): Linear(in_features=512, out_features=3840, bias=True)
            )
          )
          (lora): LoraMixin()
        )
        (transformer): BaseTransformer(
          (embedding_dropout): Dropout(p=0, inplace=False)
          (layers): ModuleList(
            (0-29): 30 x BaseTransformerLayer(
              (input_layernorm): LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
              (attention): SelfAttention(
                (query_key_value): LoraLinear(
                  (original): HackColumnParallelLinear()
                  (matrix_A): HackParameterList(
                      (0): Parameter containing: [torch.bfloat16 of size 8x1920 (cuda:0)]
                      (1): Parameter containing: [torch.bfloat16 of size 8x1920 (cuda:0)]
                      (2): Parameter containing: [torch.bfloat16 of size 8x1920 (cuda:0)]
                  )
                  (matrix_B): HackParameterList(
                      (0): Parameter containing: [torch.bfloat16 of size 1920x8 (cuda:0)]
                      (1): Parameter containing: [torch.bfloat16 of size 1920x8 (cuda:0)]
                      (2): Parameter containing: [torch.bfloat16 of size 1920x8 (cuda:0)]
                  )
                  (matrix_C): HackParameterList(
                      (0): Parameter containing: [torch.bfloat16 of size 256x1920 (cuda:0)]
                      (1): Parameter containing: [torch.bfloat16 of size 256x1920 (cuda:0)]
                      (2): Parameter containing: [torch.bfloat16 of size 256x1920 (cuda:0)]
                  )
                  (matrix_D): HackParameterList(
                      (0): Parameter containing: [torch.bfloat16 of size 1920x256 (cuda:0)]
                      (1): Parameter containing: [torch.bfloat16 of size 1920x256 (cuda:0)]
                      (2): Parameter containing: [torch.bfloat16 of size 1920x256 (cuda:0)]
                  )
                )
                (attention_dropout): Dropout(p=0, inplace=False)
                (dense): LoraLinear(
                  (original): HackRowParallelLinear()
                  (matrix_A): HackParameterList(  (0): Parameter containing: [torch.bfloat16 of size 8x1920 (cuda:0)])
                  (matrix_B): HackParameterList(  (0): Parameter containing: [torch.bfloat16 of size 1920x8 (cuda:0)])
                  (matrix_C): HackParameterList(  (0): Parameter containing: [torch.bfloat16 of size 256x1920 (cuda:0)])
                  (matrix_D): HackParameterList(  (0): Parameter containing: [torch.bfloat16 of size 1920x256 (cuda:0)])
                )
                (output_dropout): Dropout(p=0, inplace=False)
              )
              (post_attention_layernorm): LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
              (mlp): MLP(
                (activation_func): GELU(approximate='tanh')
                (dense_h_to_4h): ColumnParallelLinear()
                (dense_4h_to_h): RowParallelLinear()
                (dropout): Dropout(p=0, inplace=False)
              )
            )
          )
          (final_layernorm): LayerNorm((1920,), eps=1e-05, elementwise_affine=True)
        )
        (time_embed): Sequential(
          (0): Linear(in_features=1920, out_features=512, bias=True)
          (1): SiLU()
          (2): Linear(in_features=512, out_features=512, bias=True)
        )
      )
    )
    (denoiser): DiscreteDenoiser()
    (conditioner): GeneralConditioner(
      (embedders): ModuleList(
        (0): FrozenT5Embedder(
          (transformer): T5EncoderModel(
            (shared): Embedding(32128, 4096)
            (encoder): T5Stack(
              (embed_tokens): Embedding(32128, 4096)
              (block): ModuleList(
                (0): T5Block(
                  (layer): ModuleList(
                    (0): T5LayerSelfAttention(
                      (SelfAttention): T5Attention(
                        (q): Linear(in_features=4096, out_features=4096, bias=False)
                        (k): Linear(in_features=4096, out_features=4096, bias=False)
                        (v): Linear(in_features=4096, out_features=4096, bias=False)
                        (o): Linear(in_features=4096, out_features=4096, bias=False)
                        (relative_attention_bias): Embedding(32, 64)
                      )
                      (layer_norm): T5LayerNorm()
                      (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (1): T5LayerFF(
                      (DenseReluDense): T5DenseGatedActDense(
                        (wi_0): Linear(in_features=4096, out_features=10240, bias=False)
                        (wi_1): Linear(in_features=4096, out_features=10240, bias=False)
                        (wo): Linear(in_features=10240, out_features=4096, bias=False)
                        (dropout): Dropout(p=0.1, inplace=False)
                        (act): NewGELUActivation()
                      )
                      (layer_norm): T5LayerNorm()
                      (dropout): Dropout(p=0.1, inplace=False)
                    )
                  )
                )
                (1-23): 23 x T5Block(
                  (layer): ModuleList(
                    (0): T5LayerSelfAttention(
                      (SelfAttention): T5Attention(
                        (q): Linear(in_features=4096, out_features=4096, bias=False)
                        (k): Linear(in_features=4096, out_features=4096, bias=False)
                        (v): Linear(in_features=4096, out_features=4096, bias=False)
                        (o): Linear(in_features=4096, out_features=4096, bias=False)
                      )
                      (layer_norm): T5LayerNorm()
                      (dropout): Dropout(p=0.1, inplace=False)
                    )
                    (1): T5LayerFF(
                      (DenseReluDense): T5DenseGatedActDense(
                        (wi_0): Linear(in_features=4096, out_features=10240, bias=False)
                        (wi_1): Linear(in_features=4096, out_features=10240, bias=False)
                        (wo): Linear(in_features=10240, out_features=4096, bias=False)
                        (dropout): Dropout(p=0.1, inplace=False)
                        (act): NewGELUActivation()
                      )
                      (layer_norm): T5LayerNorm()
                      (dropout): Dropout(p=0.1, inplace=False)
                    )
                  )
                )
              )
              (final_layer_norm): T5LayerNorm()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
    )
    (first_stage_model): VideoAutoencoderInferenceWrapper(
      (encoder): ContextParallelEncoder3D(
        (conv_in): ContextParallelCausalConv3d(
          (conv): SafeConv3d(3, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        )
        (down): ModuleList(
          (0): Module(
            (block): ModuleList(
              (0-2): 3 x ContextParallelResnetBlock3D(
                (norm1): ContextParallelGroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): ContextParallelGroupNorm(32, 128, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
              )
            )
            (attn): ModuleList()
            (downsample): DownSample3D(
              (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2))
            )
          )
          (1): Module(
            (block): ModuleList(
              (0): ContextParallelResnetBlock3D(
                (norm1): ContextParallelGroupNorm(32, 128, eps=1e-06, affine=True)
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): ContextParallelGroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (nin_shortcut): SafeConv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
              (1-2): 2 x ContextParallelResnetBlock3D(
                (norm1): ContextParallelGroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): ContextParallelGroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
              )
            )
            (attn): ModuleList()
            (downsample): DownSample3D(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
            )
          )
          (2): Module(
            (block): ModuleList(
              (0-2): 3 x ContextParallelResnetBlock3D(
                (norm1): ContextParallelGroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): ContextParallelGroupNorm(32, 256, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
              )
            )
            (attn): ModuleList()
            (downsample): DownSample3D(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
            )
          )
          (3): Module(
            (block): ModuleList(
              (0): ContextParallelResnetBlock3D(
                (norm1): ContextParallelGroupNorm(32, 256, eps=1e-06, affine=True)
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): ContextParallelGroupNorm(32, 512, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (nin_shortcut): SafeConv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
              (1-2): 2 x ContextParallelResnetBlock3D(
                (norm1): ContextParallelGroupNorm(32, 512, eps=1e-06, affine=True)
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): ContextParallelGroupNorm(32, 512, eps=1e-06, affine=True)
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
              )
            )
            (attn): ModuleList()
          )
        )
        (mid): Module(
          (block_1): ContextParallelResnetBlock3D(
            (norm1): ContextParallelGroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): ContextParallelCausalConv3d(
              (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): ContextParallelGroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): ContextParallelCausalConv3d(
              (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
          )
          (block_2): ContextParallelResnetBlock3D(
            (norm1): ContextParallelGroupNorm(32, 512, eps=1e-06, affine=True)
            (conv1): ContextParallelCausalConv3d(
              (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): ContextParallelGroupNorm(32, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): ContextParallelCausalConv3d(
              (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
          )
        )
        (norm_out): ContextParallelGroupNorm(32, 512, eps=1e-06, affine=True)
        (conv_out): ContextParallelCausalConv3d(
          (conv): SafeConv3d(512, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        )
      )
      (decoder): ContextParallelDecoder3D(
        (conv_in): ContextParallelCausalConv3d(
          (conv): SafeConv3d(16, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        )
        (mid): Module(
          (block_1): ContextParallelResnetBlock3D(
            (norm1): SpatialNorm3D(
              (norm_layer): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv_y): ContextParallelCausalConv3d(
                (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
              (conv_b): ContextParallelCausalConv3d(
                (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
            )
            (conv1): ContextParallelCausalConv3d(
              (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): SpatialNorm3D(
              (norm_layer): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv_y): ContextParallelCausalConv3d(
                (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
              (conv_b): ContextParallelCausalConv3d(
                (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): ContextParallelCausalConv3d(
              (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
          )
          (block_2): ContextParallelResnetBlock3D(
            (norm1): SpatialNorm3D(
              (norm_layer): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv_y): ContextParallelCausalConv3d(
                (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
              (conv_b): ContextParallelCausalConv3d(
                (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
            )
            (conv1): ContextParallelCausalConv3d(
              (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): SpatialNorm3D(
              (norm_layer): GroupNorm(32, 512, eps=1e-06, affine=True)
              (conv_y): ContextParallelCausalConv3d(
                (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
              (conv_b): ContextParallelCausalConv3d(
                (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
            )
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): ContextParallelCausalConv3d(
              (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
          )
        )
        (up): ModuleList(
          (0): Module(
            (block): ModuleList(
              (0): ContextParallelResnetBlock3D(
                (norm1): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 256, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 128, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (nin_shortcut): SafeConv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
              (1-3): 3 x ContextParallelResnetBlock3D(
                (norm1): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 128, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 128, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
              )
            )
            (attn): ModuleList()
          )
          (1): Module(
            (block): ModuleList(
              (0-3): 4 x ContextParallelResnetBlock3D(
                (norm1): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 256, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 256, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
              )
            )
            (attn): ModuleList()
            (upsample): Upsample3D(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (2): Module(
            (block): ModuleList(
              (0): ContextParallelResnetBlock3D(
                (norm1): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 512, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 256, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (nin_shortcut): SafeConv3d(512, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
              )
              (1-3): 3 x ContextParallelResnetBlock3D(
                (norm1): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 256, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 256, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
              )
            )
            (attn): ModuleList()
            (upsample): Upsample3D(
              (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
          (3): Module(
            (block): ModuleList(
              (0-3): 4 x ContextParallelResnetBlock3D(
                (norm1): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 512, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (conv1): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
                (norm2): SpatialNorm3D(
                  (norm_layer): GroupNorm(32, 512, eps=1e-06, affine=True)
                  (conv_y): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                  (conv_b): ContextParallelCausalConv3d(
                    (conv): SafeConv3d(16, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
                  )
                )
                (dropout): Dropout(p=0.0, inplace=False)
                (conv2): ContextParallelCausalConv3d(
                  (conv): SafeConv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
                )
              )
            )
            (attn): ModuleList()
            (upsample): Upsample3D(
              (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
          )
        )
        (norm_out): SpatialNorm3D(
          (norm_layer): GroupNorm(32, 128, eps=1e-06, affine=True)
          (conv_y): ContextParallelCausalConv3d(
            (conv): SafeConv3d(16, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
          )
          (conv_b): ContextParallelCausalConv3d(
            (conv): SafeConv3d(16, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1))
          )
        )
        (conv_out): ContextParallelCausalConv3d(
          (conv): SafeConv3d(128, 3, kernel_size=(3, 3, 3), stride=(1, 1, 1))
        )
      )
      (loss): Identity()
      (regularization): DiagonalGaussianRegularizer()
    )
    (loss_fn): VideoDiffusionLoss()
  )
)