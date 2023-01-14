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
    figure;
    imshow(region,[]);

    cell = icut(region, s);
    cell = bwareaopen(cell, 600,4);
    L1 = bwlabel(cell,4);
    purple = label2rgb(L1,'jet','k','shuffle');
    figure, imshow(purple);

    figure,
    imshow(AR);
    hold on
    stats = regionprops(L1,'BoundingBox','Centroid', 'Area');
    G=cat(1,stats.Centroid);
    scatter(G(:,1), G(:,2), 30, 'red', 'filled');
    
%----------------------------Evaluation----------------------------------------
%     addpath('G:\software\MATLAB2022a\toolbox\jsonlab-master\jsonlab-master')
%     label = 'CM310-1(25)6-(400X).json';
%     label_json = loadjson(label);
%     label_point = cat(1,label_json.shapes.points);
%     a = label_point;
%     b = G;
%     [TP, FN, FP] = quantity(a,b);
%     
%     figure,
%     imshow(AR);
%     hold on
%     scatter(uint16(TP(:,1)), uint16(TP(:,2)), 40, 'green', 'filled');
%     scatter(FP(:,1), FP(:,2), 40, 'red', 'filled');
%     scatter(FN(:,1), FN(:,2), 40, 'yellow', 'filled');
%     recall = size(TP,1)/(size(TP,1)+size(FN,1));
%     precision = size(TP,1)/(size(TP,1)+size(FP,1));
%     F_score = 2*recall*precision/(recall + precision);
end