function [TP, FN, FP] = quantity(a, b)
m = size(a,1);

for i = 1:m
    minus = b-a(i,:);
    dis= sqrt(minus(:,1).^2 + minus(:,2).^2);
    idx = find(dis <= 40);
    if ~isempty(idx)
        TP_idx = find(dis == min(dis(idx)));
        TP(i,:) = b(TP_idx,:);
        b(TP_idx,:) = [];
    else
        TP(i,1:2) = NaN;
    end
    
end
FN = a.*isnan(TP);
FN(find(FN(:,1)==0),:)=[];
FP = b;
TP = TP(all(~isnan(TP),2),:);
end