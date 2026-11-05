#import "../helpers.typ": guideline, fig-placeholder

= Introduction

// Cường có sửa ý thôi, còn câu văn thì AI viết, Huyền paraphase lại cho dễ hiểu như đoạn trước của Huyền đi.

// Đoạn 2, dẫn dắt 2 paper thì, theo Cường đọc, paper đầu tiên có các hạn chế là : Làm trên ảnh lớn, reverse noising không nhảy cóc được như forward noise -> paper thứ 2, làm trên ảnh đã nén bằng ae và có vài định hướng noise,... 
// H: này có viết từ đoạn In denoising... although...to reduce this cost rùi á

// à đoạn cost đó Cường thấy cũng hơi hơi lạ,

// Đoạn thứ 3: bài mình là thực nghiệm lại trên ảnh bé 28 * 28, còn lại mấy ý so sánh, giải thích forward, reverse,... ok òi

// Đoạn: The rest of ... chắc bỏ đi cũng được cho đỡ dài

// À thêm là khom in đậm nghiêng nheee, Huyền lấy intro.md của Cường gửi trong mess òi xem thử đi
// ủa mà sao ko được in đậm nghiêng dị

// In đậm thì phải note ra tại sao á. Tai thường in đậm nó là số liệu tốt, hoặc error, và nên note ra để reviewer hiểu chứ không được mặc định á. Còn in nghiêm thì trích code, hoặc trích văn của người khác ra mới làm dị. okok

// đã sửa

Artificial intelligence (AI) image synthesis is the process of generating new images by learning patterns from existing data. It is widely used in areas such as digital content creation, image editing, and visual design.
However, generating realistic, diverse, and high-resolution images is still a challenging problem. Several generative approaches have been developed to address this task. For example, Generative Adversarial Networks (GANs) can generate images quickly, but their training process can be unstable and may suffer from _mode collapse_. Autoregressive models generate images by modeling conditional probabilities, although their sequential generation process can be slow when dealing with high-dimensional data @goodfellow2014gan @rombach2022ldm.


Diffusion Models use a different method to generate images. Instead of creating an image directly, they gradually add noise to training images and then learn how to remove that noise. Once the model has learned this process, it can generate new images starting from random noise. In Denoising Diffusion Probabilistic Models (DDPM), Ho et al. @ho2020ddpm show that a neural network can learn to predict the noise added to an image using a simplified training objective. Although DDPM can produce high-quality images, it requires many denoising steps, which makes image generation computationally expensive. This problem becomes more noticeable when generating high-resolution images. To reduce this cost, Rombach et al. @rombach2022ldm propose Latent Diffusion Models (LDM), which perform the diffusion process in a compressed latent space instead of directly on image pixels.

The report aims to investigate how learning to remove noise can enable a model to generate new images. We also also examines why performing diffusion in latent space can reduce computational costs while still preserving important visual information. Beyond theoretical analysis, we implement both models and conduct an empirical comparison on $28 times 28$ images to evaluate their practical trade-offs. Specifically, the report aims to: (i) explain the difference between discriminative and generative models and compare common generative approaches; (ii) describe how forward diffusion, reverse diffusion, and noise prediction work in DDPM; (iii) illustrate these ideas through a one-dimensional numerical example and images with different noise levels; and (iv) explain how LDM uses autoencoders and conditioning mechanisms, and experimentally evaluate and compare the computational efficiency and generation behavior of DDPM and LDM on a $28 times 28$ image benchmark, including the trade-offs between image quality, computational efficiency, and flexibility.




