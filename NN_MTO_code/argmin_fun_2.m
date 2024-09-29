function [ p ] = argmin_fun_2( fn1, fn2 )
    n1 = length(fn1);
    n2 = length(fn2);
    p_min = inf*ones(1,n1);
    p = -1*ones(1,n1);
    for i = 1:n1
        for j = 1:n2
            dif = abs(fn1(i)-fn2(j));
            if dif < p_min(i)
                p_min(i) = dif;
                p(i) = j;
            end
        end
    end
end

