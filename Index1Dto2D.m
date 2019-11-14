function [x y] = Index1Dto2D(Index1D, H, W)
% 将1D坐标转换为2D坐标
x = mod(Index1D,H);
y = (Index1D-x)/H + 1;
if (x==0)
    y = y - 1;
    x = H;
end