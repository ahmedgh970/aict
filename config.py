class ConfigGa:
  embed_dim = [128, 192, 256, 320]
  embed_out_dim = [192, 256, 320, None]
  depths = [2, 2, 6, 2]
  head_dim = [32, 32, 32, 32]
  window_size = [8, 8, 8, 8]
  num_layers = len(depths)

class ConfigHa:
  embed_dim = [192, 192]
  embed_out_dim = [192, None]
  depths = [5, 1]
  head_dim = [32, 32]
  window_size = [4, 4]
  num_layers = len(depths)

class ConfigHs:
  embed_dim = [192, 192]
  embed_out_dim = [192, int(2*320)]
  depths = [1, 5]
  head_dim = [32, 32]
  window_size = [4, 4]
  num_layers = len(depths)


class ConfigST:
  embed_dim     = [32, 64, 96, 128, 160, 192, 192, 192, 192, 192]
  embed_out_dim = [32, 64, 96, 128, 160, 192, 192, 192, 192, 192]
  depths = 2
  head_dim = [4, 8, 16, 16, 32, 32, 32, 32, 32, 32]
  window_size = 4
  
class ConfigGs:
  embed_dim = [320, 256, 192, 128]
  embed_out_dim = [256, 192, 128, 3]
  depths = [2, 6, 2, 2]
  head_dim = [32, 32, 32, 32]
  window_size = [8, 8, 8, 8]
  num_layers = len(depths)

class ConfigChARM:
  depths_conv0 = depths_swin0 = [64, 64, 85, 106, 128, 149, 170, 192, 213, 234]
  depths_conv1 = depths_swin1 = [32, 32, 42, 53, 64, 74, 85, 96, 106, 117]

