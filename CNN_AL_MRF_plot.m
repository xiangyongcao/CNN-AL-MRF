function CNN_AL_MRF_plot(labels, size, dir)

labels = reshape(labels, size);
im = zeros([size, 3]);
colorbar = [[0 0 0]; ...%% 0. Black: background
    [0 0 255]; ...%% 1. Blue
    [0 255 0]; ...%% 2. Beans?
    [255 0 0]; ...%% 3. Beet?
    [0 102 205]; ...%% 4. Ç³À¶£¿
    [0 204 102]; ...%% 5. Maize
    [255 128 0]; ...%% 6. Potato
    [102 205 0]; ...%% 7. ?
    [102 0 204]; ...%% 8. Lucerne
    [204 0 102]; ...%% 9. Rapeseed
    [204 102 255]; ...%% 10. Peas?
    [251 232 45]; ...%% 11. Wheat
    [138 42 166]; ...%% 12. Fruit?
    [120 178 215]; ...%% 13. Barley
    [204 255 204]; ...%% 14. Flax?
    [255 204 204]; ...%% 15. Grass?
    [40 210 180]; ...%% 16. ?
    ]/255;
for i=1:size(1)
    for j=1:size(2)
        im(i, j, :) = colorbar(labels(i, j)+1, :);
    end
end
figure, imshow(im);
save(strcat(dir,'.mat'), 'im');
imwrite(im, strcat(dir, '.jpg'));