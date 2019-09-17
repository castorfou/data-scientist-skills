A = [1 2; 3 4; 5 6];
B = [1 2 3; 4 5 6];
C=A*B;
C=B'+A;
#C=A'*B;
#C=B+A;

A=eye(10)
x=[1;2;3;4;5;6;7;8;9;10]
D=A.*x

v = zeros(10, 1);
for i = 1:10
  for j = 1:10
    v(i) = v(i) + A(i, j) * x(j);
  end
end

v=[1;2;3;4;5;6;7]
w=[1;2;3;4;5;6;7]
z = 0;
for i = 1:7
  z = z + v(i) * w(i)
end

z2=sum(v.*w)
z3=w'*v
z4=v*w
z5=w*v


