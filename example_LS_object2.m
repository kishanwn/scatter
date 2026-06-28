

for i = 1:1 % 5000

h = 2.0^(-10);
omega = 40.0;

% size of box
a = 0.5;
b = 0.5; % 0.25;

% discretization
x = -a/2:h:a/2-h;
y = -b/2:h:b/2-h;



n = length(x) 
m = length(y)

X = repmat(x', 1, m);
Y = repmat(y, n,1);


rand_noise_nu = randn(n,m)/10  

% the heterogeneity
nu = @(x,y) 1.5*exp(-160*(x.^2 + y.^2));   % -0.3*exp(-320*(x.^2 + y.^2)).*(abs(x)<0.48).*(abs(y)<0.48);

% the slowness squared
M = 1 + nu(X,Y);
M =  nu(X,Y);
# max(max(M))
% plotting the slowness squared

% hold on;
% figure(1); clf();
% title("slowness squared")
% imagesc(M)
% hold off;
% imwrite(M , "my_output_image3.jpg");

 
LS = LippmannSchwinger(x,y,omega,nu,a, rand_noise_nu)

% LS.nu

u_inc = exp(omega*1i*X);
rhsDual = -omega^2*nu(X,Y).*u_inc; 

sigma = LS\rhsDual(:);

% computing the wavefield
u = LS.apply_Green(sigma); 

% plotting the scattered wavefield
% hold on;
% figure(2); clf();
% imagesc(real(reshape(u, n, m)));
% hold off;

% max(max(real(u)))
% imwrite (real(reshape(u, n, m)), "my_output_image2.jpg");
% plotting the total wavefield

% figure(1)
% hold on
% clf();
% imagesc(real(reshape(u+u_inc, n, m)))
% imwrite (real(reshape(u+u_inc, n, m)), "my_output_imag%e.jpg");
% hold off


qname = ["x.mat"]
save('-v7', qname, "x")

qname = ["y.mat"]
save('-v7', qname, "y")


%nu_ = LS.nu;
%qname = ["result/q" , num2str(i) , ".mat"]
%save('-v7', qname, "nu_")

%realu = real(u);
%qname = ["result/real" , num2str(i) , ".mat"]
%save('-v7', qname, "realu")

%imagu = imag(u);
%qname = ["result/imag" , num2str(i) , ".mat"]
%save('-v7', qname, "imagu")

end