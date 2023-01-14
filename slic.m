function region = slic(A)
    I = rgb2lab(A);
    H = rgb2hsv(A);
    [m,n,z] = size(A);
    [L, N] = superpixels(I, uint16(floor(m*n/1500)), 'IsInputLab', true);
    figure
    BW = boundarymask(L);
    imshow(imoverlay(A, BW, 'cyan'))
    centers1 = 68.25;
    centers2 = 21.18;
    centers3 = 20;
    centers4 = 2.76;
    centers5 = 3.07;   

    outputImage = zeros(m,n,'single');
    idx = label2idx(L);
    numRows = size(I,1);
    numCols = size(I,2);
    
    a_mean = regionprops(L, I(:,:,2), 'MeanIntensity');
    a_p=cat(1, a_mean.MeanIntensity);
    [k_idx, C] = kmeans(a_p,5,  'start', [centers1; centers2; centers3; centers4; centers5]);
    for labelVal = 1:N
        l_Idx = idx{labelVal};
        a_Idx = idx{labelVal}+numRows*numCols;
        b_Idx = idx{labelVal}+2*numRows*numCols;
        outputImage(l_Idx) = k_idx(labelVal);
    end    
    outputImage(outputImage ~= 1) = 0;
    region = logical(outputImage);

    outputImage2 = zeros(size(A),'single');
     for labelVal2 = 1:N
        l_Idx2 = idx{labelVal2};
        a_Idx2 = idx{labelVal2}+numRows*numCols;
        b_Idx2 = idx{labelVal2}+2*numRows*numCols;
        outputImage2(l_Idx2) = mean(I(l_Idx2));
        outputImage2(a_Idx2) = mean(I(a_Idx2));
        outputImage2(b_Idx2) = mean(I(b_Idx2));
    end    
     figure
     imshow(uint8(outputImage2(:,:,2)),[])

end