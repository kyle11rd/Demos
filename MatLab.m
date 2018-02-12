clear
clc

op=fopen('C:\Test\demo.txt','r');
txt=textscan(op,'%s%d%f%*[^\n]','delimiter','\t','headerlines',1);
fclose(op);

a = strfind(txt{1}{1},'abc');

b = find(txt{2}==9,1,'last');

wp=fopen('C:\Test\out.txt','w');
for i=1:length(txt{1})
    fprintf(wp,'%s\t%d\t%.2f\n',txt{1}{i},txt{2}(i),txt{3}(i));
end
fclose('all');
