from mllm.dataset import TokenFormatter

tk = TokenFormatter()

ret = tk.extract('ASSISTANT: Answer: <b_st> <bin_432> <bin_511> <bin_565> <bin_816> <b_ed>  .</s><unk>')
print(ret)
