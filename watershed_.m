
clc
close all
clear 
img_Folder = 'image\';
mask_Forder = 'mask\';
img_List = dir(fullfile(img_Folder, '*.jpg')); 
mask_List = dir(fullfile(mask_Forder, '*.mat'));
for i = 2: length(img_List)   
    img_path = [img_Folder, img_List(i).name];
    mask_path = [mask_Forder, mask_List(i).name];
    AR = imread(img_path);
    A = color_norm(AR);
    load (mask_path)
    region = slic(A);
    s = 1.5;
    region(h1 == 0) = 0;
    bw2 = ~bwareaopen(~region, 10);
    imshow(bw2);
    D = -bwdist(~region);
    imshow(D,[]);
    Ld = watershed(D);
    imshow(label2rgb(Ld));
    bw2 =  region;
    bw2(Ld == 0) = 0;
    imshow(bw2);
    mask = imextendedmin(D,2);
    imshowpair( region,mask,'blend');
    D2 = imimposemin(D,mask);
    Ld2 = watershed(D2);
    bw3 =  region;
    bw3(Ld2 == 0) = 0;
    imshow(bw3);
end