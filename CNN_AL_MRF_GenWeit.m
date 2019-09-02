function CNN_AL_MRF_GenWeit(data, DirHor, DirVer)

BandInd = data.IndBand;
%BandInd = [12,60,180];
if ~isfield(data, 'F_norm')
    Feature = data.F;
    % normalize
    for b=1:length(BandInd)
        Feature(:,:,BandInd(b)) = (Feature(:,:,BandInd(b))-min(min(Feature(:,:,BandInd(b)))))./...
            (max(max(Feature(:,:,BandInd(b))))-min(min(Feature(:,:,BandInd(b)))));
    end
end
Feature = data.F_norm;
HSI_rgb = im2uint8(Feature(:,:,BandInd));
% figure, imshow(HSI_rgb)

H = size(HSI_rgb, 1);
W = size(HSI_rgb, 2);

R = flipud(HSI_rgb(:,:,1)); R = R(1:H,:);R = R(:,1:W);
G = flipud(HSI_rgb(:,:,2)); G = G(1:H,:);G = G(:,1:W);
B = flipud(HSI_rgb(:,:,3)); B = B(1:H,:);B = B(:,1:W);

for i = 2 : H
    for j = 2 : W
        Dij_left(i-1,j-1) = (abs(R(i,j) - R(i-1,j)))^2;
        Dij_up(i-1,j-1) = (abs(R(i,j) - R(i,j-1)))^2;
    end
end
var_left = std(double(Dij_left(:)));
var_up = std(double(Dij_up(:)));
HorzWeight1 = exp(-double(Dij_left)/(2*var_left));
VertWeight1 = exp(-double(Dij_up)/(2*var_up));

for i = 2 : H
    for j = 2 : W
        Dij_left(i-1,j-1) = (abs(G(i,j) - G(i-1,j)))^2;
        Dij_up(i-1,j-1) = (abs(G(i,j) - G(i,j-1)))^2;
    end
end
var_left = std(double(Dij_left(:)));
var_up = std(double(Dij_up(:)));
HorzWeight2 = exp(-double(Dij_left)/(2*var_left));
VertWeight2 = exp(-double(Dij_up)/(2*var_up));

for i = 2 : H
    for j = 2 : W
        Dij_left(i-1,j-1) = (abs(B(i,j) - B(i-1,j)))^2;
        Dij_up(i-1,j-1) = (abs(B(i,j) - B(i,j-1)))^2;
    end
end
var_left = std(double(Dij_left(:)));
var_up = std(double(Dij_up(:)));
HorzWeight3 = exp(-double(Dij_left)/(2*var_left));
VertWeight3 = exp(-double(Dij_up)/(2*var_up));

HorzWeight = real(HorzWeight1+HorzWeight2+HorzWeight3)/3;
VertWeight = real(VertWeight1+VertWeight2+VertWeight3)/3;

% figure;
% mesh(HorzWeight);
% title('HorzWeight');
% axis([1 W 1 H]);

% figure;
% mesh(VertWeight);
% title('VertWeight');
% axis([1 W 1 H]);

save(DirHor, 'HorzWeight', '-ascii');
save(DirVer, 'VertWeight', '-ascii');
