class resnet_hope_blocks(nn.Module):

    def __init__(self, device, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(resnet_hope_blocks, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.device = device

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size), device=device),2)) # 64x48 = 3072
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size*self.f_size), device=device),2)) # 48

        self.w2 = Parameter(torch.fmod(torch.randn(((17), self.in_size*self.z_dim), device=device),2)) # 17x1024 = 17408
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim), device=device),2)) # 1024

        self.w3 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim, self.in_size*self.z_dim//8), device=device),2)) # 1024x128 = 131072
        self.b3 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim//8), device=device),2)) # 128

        self.w4 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim//8, self.in_size*self.z_dim), device=device),2)) # 128x1024 = 131072
        self.b4 = Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim), device=device),2)) # 1024


        # total params = 3072 + 48 + 17408 + 1024 + 131072 + 128 + 131072 + 1024 = 484848

    def forward(self, z):

        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = torch.tanh(h_in)  # Activation here
        h_in = torch.matmul(h_in, self.w3) + self.b3
        h_in = torch.tanh(h_in)
        h_in = torch.matmul(h_in, self.w4) + self.b4
        h_in = torch.tanh(h_in)
        #introduce w2, w3
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        h_final = torch.tanh(h_final) # added tanh activation 

        kernel = h_final.view(self.out_size, self.in_size, self.f_size)

        return kernel
