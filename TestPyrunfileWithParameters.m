% my_script4pyrunfile.py

% 定义要传递的字符串变量
str_var = "Hello from MATLAB";

% 使用 pyrunfile 运行 Python 脚本并传递变量
pyrunfile("my_script4pyrunfile.py", "data", str_var);