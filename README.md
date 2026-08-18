# PTS
(Pytorch too slow)

# Introduction

This is a high performance kernel optimization for training LLMs. Pytorch is (relatively) slow and cannot be used for resource constrained instances, so this is made to combat that. The first major use case is what is the maximum number of params I can fit on 10 GB of VRAM. I will be trying various memory optimization techniques. The main library will be in PTS folder, everything else can be ignored. Bench marking scripts will be vibe coded because I cannot be bothered to write those myself.