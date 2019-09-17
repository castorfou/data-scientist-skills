indices=1:10;
indices
for i=indices,
  disp(i);
endfor
v=zeros(10,1)
for i=indices,
  v(i)=2^i;
endfor
v
i=1;
while i<=5,
  v(i)=100;
  i=i+1;
endwhile
v
i=1;
while true,
  v(i) = 999;
  i=i+1;
  if i==6,
    break;
  endif
endwhile
v
v(1)=2;
if v(1)==1,
  disp('value is one')
elseif v(1) ==2,
  disp('value is two')
else
  disp('the value is not one or two')
  
endif

squareThisNumber(10)

X=[1 1; 1 2; 1 3]
y=[1;2;3]
theta=[0;1];
J=costFunctionJ(X,y,theta)
plot(X,'+')