function [sequence, pss]= NN_MTO(Tasks, maxfes, problem_index,pars)%���������Ⱥ��С
FE = 0;
ntask = length(Tasks);
dims = zeros(1, ntask);
for i = 1:ntask
    dims(i) = Tasks(i).dim;
end

n = 50*ones(1,ntask);
n_max = 70;
n_min = 30;
his_best_fit = zeros(1,ntask);
update_state = zeros(1,ntask);
stag_state = zeros(1,ntask);


% result record
sequence = [];
pss = [];
save_gap = maxfes/10;
cur_gap = 0;

%% initilization
pop_list = {};
fit_list = {};

for i = 1:ntask
    pop = rand(n(i), dims(i));
    pop_list{i} = pop;
    fit = zeros(1, n(i));
    for j = 1:n(i)
        [fit(j),~]=fnceval(Tasks(i),pop(j, :));
        FE = FE + 1;
    end
    fit_list{i} = fit;
    his_best_fit(i) = min(fit);
end

%% ����
gen = 0;
while FE < maxfes

    if FE >= cur_gap
    data = [];
    for t = 1:ntask
        data = [data, min(fit_list{t})];
    end
    sequence = [sequence; data];
    pss = [pss;n(1),n(2)];
    disp(['fes:', num2str(FE), ' gen:', num2str(gen), ' fit1: ', num2str(data(1)), ' fit2: ', num2str(data(2)),' n:',num2str(n(1)),' ',num2str(n(2))]);
    cur_gap = cur_gap + save_gap;
    end

    gen = gen + 1;

    %% ѵ��������
    NN_G = pars.NN_G;
    if mod(gen,NN_G)==1
        try
            filename = strcat('NN\NN_p', num2str(problem_index), '_g',num2str(gen),'.mat');
            load(filename);
            net = saved_net;
            saved_net=[];
        catch
            net = cell(ntask,ntask);
            for i = 1:ntask
                for j = 1:ntask
                    net{i,j} = cell(1,dims(j));% ��ʼ��������
                end
            end
            for i = 1:ntask-1
                for j = i+1:ntask
                    if length(fit_list{i}) < length(fit_list{j})
                        all_length{i} = length(fit_list{i});
                        all_length{j} = length(fit_list{i});
                   else
                        all_length{i} = length(fit_list{j});
                        all_length{j} = length(fit_list{j});
                    end
                    [fit_i,sort_index] = sort(fit_list{i},'ascend');
                    pop_i = pop_list{i}(sort_index,:);
                    fmax_i = fit_i(end);
                    fmin_i = fit_i(1);
                    [fit_j,sort_index] = sort(fit_list{j},'ascend');
                    pop_j = pop_list{j}(sort_index,:);
                    fmax_j = fit_j(end);
                    fmin_j = fit_j(1);
                    %j->i��������ѧϰ
                    normalized_fit_i = [];
                    normalized_fit_j = [];
                    for k = 1:all_length{i}
                        normalized_fit_i(k) = (fit_i(k) - fmin_i)/(fmax_i-fmin_i+1E-10);
                    end
                    for k = 1:all_length{j}
                        normalized_fit_j(k) = (fit_j(k) - fmin_j)/(fmax_j-fmin_j+1E-10);
                    end
                    p_index = argmin_fun_2(normalized_fit_i, normalized_fit_j);% error
                    output = pop_i(1:all_length{i},:);
                    input = pop_j(p_index,:);

                    intput_num = dims(j);
                    hidden_num = ceil(max( size(output,1) /((dims(i)+1)*10), 1));
                    output_num = 1;

                    for d = 1:dims(i)
                        output_d = output(:,d)';
                        net{j,i}{d} = newff(input', output_d, hidden_num, {'tansig','purelin'}, 'trainlm');%���ܺ���Ĭ��ֵΪmse
                        net{j,i}{d}.trainParam.epochs = 30;                 %ѵ������
                        net{j,i}{d}.trainParam.lr = 0.01;                   %ѧϰ����
                        net{j,i}{d}.trainParam.min_grad = 1e-20;            %��С�ݶ�����
                        net{j,i}{d}.trainParam.goal = 0.00001;              %ѵ��Ŀ����С��� %TODO �и�ָ����ܱȵ��������ȵ��ע���Ƿ�Ҫ�޸�
                        net{j,i}{d}.divideFcn = '';                         %��������������6�ε�����û�仯����Ĭ����ֹѵ����Ϊ���ó���������У��ô�����ȡ����������
                        net{j,i}{d}.trainParam.showWindow = false;          %ȡ��ѵ������
                        net{j,i}{d} = train(net{j,i}{d}, input', output_d);  %��ʼѵ��  ע��һ��Ϊ��������ôһ�����Ϊһ��
                    end

                    %i->j��������ѧϰ 
                    p_index = argmin_fun_2(normalized_fit_j, normalized_fit_i);%error
                    output = pop_j(1:all_length{j},:);
                    input = pop_i(p_index,:);

                    intput_num = dims(i);
                    hidden_num = ceil(max( size(output,1) /((dims(j)+1)*10), 1));
                    output_num = 1;

                    for d = 1:dims(j)
                        output_d = output(:,d)';
                        net{i,j}{d} = newff(input', output_d, hidden_num, {'tansig','purelin'}, 'trainlm');%���ܺ���Ĭ��ֵΪmse
                        net{i,j}{d}.trainParam.epochs = 30;                 %ѵ������
                        net{i,j}{d}.trainParam.lr = 0.01;                   %ѧϰ����
                        net{i,j}{d}.trainParam.min_grad = 1e-20;            %��С�ݶ�����
                        net{i,j}{d}.trainParam.goal = 0.00001;              %ѵ��Ŀ����С���
                        net{i,j}{d}.divideFcn = '';                         %��������������6�ε�����û�仯����Ĭ����ֹѵ����Ϊ���ó���������У��ô�����ȡ����������
                        net{i,j}{d}.trainParam.showWindow = false;          %ȡ��ѵ������
                        net{i,j}{d} = train(net{i,j}{d}, input', output_d);  %��ʼѵ��
                    end
                end
            end
        end
    end
    kt_g = pars.kt_g;
    for t = 1:ntask
        s = pars.s;% %��ȡ����
        ind{t} = [];ind_fit{t} = [];
        if gen>=1 &&mod(gen,kt_g)==1 
            source_t = randi(ntask);
            while(source_t == t)
                source_t = randi(ntask);
            end
            inorder = randperm(n(3-t));
            source_pop = pop_list{source_t}(inorder(1:s),:);
            %��ѵ���õ����������Ԥ��
            for d = 1:dims(t)
                predict = sim(net{source_t, t}{d}, source_pop');
                ind{t} = [ind{t}, predict'];  
            end
            ind{t}(ind{t}>1)=1; 
            ind{t}(ind{t}<0)=0;
            for i=1:s
                [ind_fit{t}(i)] = fnceval(Tasks(t), ind{t}(i,:));
                FE=FE+1;
            end

            parent_pop{t} = [pop_list{t}; ind{t}];
            child_pop{t} = [pop_list{t}; ind{t}];
            child_fit{t} = [fit_list{t}, ind_fit{t}];
        else
            parent_pop{t} = [pop_list{t};];
            child_pop{t} = [pop_list{t};];
            child_fit{t} = [fit_list{t}];
        end

    end

    for t = 1:ntask
        inorder1 = randperm(n(t));
        for i = 1:n(t)
            siz = size(parent_pop{t}, 1);
            if 1
                r1 = randi(n(t));
                r2 = randi(siz);
                r3 = randi(siz);
                while r1 == r2 || r2 == r3 || r1 == r3 || r1 == i || r2 == i || r3 == i
                    r1 = randi(siz);
                    r2 = randi(siz);
                    r3 = randi(siz);
                end
                p1 = parent_pop{t}(r1, :);
                p2 = parent_pop{t}(r2, :);
                p3 = parent_pop{t}(r3, :);
                v = p1 + 0.5*(p2 - p3);
                u = parent_pop{t}(i, :);
                jrand = randi(dims(t));
                for j = 1:dims(t)
                    if rand()<0.6 || jrand ==j
                        u(j) = v(j);
                    end
                end
            end
            u = max(0, min(1, u));
            [ufit] = fnceval(Tasks(t), u);
            child_pop{t} = [child_pop{t}; u];
            child_fit{t} = [child_fit{t}, ufit];
            FE = FE + 1;
        end
        [~, sort_i] = sort(child_fit{t}, 'ascend');
        pop_list{t} = child_pop{t}(sort_i(1:n(t)),:);
        fit_list{t} = child_fit{t}(sort_i(1:n(t)));
%%------------�����˿�--------------------------------
        cur_best_fit = min(fit_list{t});
        if cur_best_fit < his_best_fit(t)
            his_best_fit(t) = min(fit_list{t});
            update_state(t) = update_state(t)+1;
            stag_state(t) = 0;
        else
            stag_state(t) = stag_state(t)+1;
            update_state(t) = 0;
        end
        u=[];
        if update_state(t) >= 1 && n(t) > n_min
            pop_list{t}(end,:) = [];
            fit_list{t}(end) = [];
            n(t) = n(t)-1;
            update_state(t) = 0;  
        elseif stag_state(t) >= 5 && n(t) < n_max
            if dims(3-t)>=dims(t)
                u = pop_list{3-t}(randi(10),1:dims(t));
            else
                u = pop_list{3-t}(randi(10),:);
                for d = dims(3-t)+1:dims(t)
                     predict = sim(net{3-t, t}{d}, pop_list{3-t}(randi(n(3-t)),:)');
                     u = [u, predict];  
                end
                u = max(0, min(1, u));
            end
            [ufit] = fnceval(Tasks(t), u);
            pop_list{t} = [pop_list{t}; u];
            fit_list{t} = [fit_list{t}, ufit];
            n(t) = n(t)+1;
            stag_state(t) = 0;  
        end
    end
end

%%
data = [];
for t = 1:ntask
    data = [data, min(fit_list{t})];
end
sequence = [sequence; data];
pss = [pss;n(1),n(2)];
disp(['fes:', num2str(FE), ' gen:', num2str(gen), ' fit1: ', num2str(data(1)), ' fit2: ', num2str(data(2))]);
end