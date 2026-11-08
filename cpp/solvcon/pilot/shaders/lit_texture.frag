#version 440

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 v_texcoord;
layout(binding = 1) uniform sampler2D baseTexture;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    vec4 color;
    vec4 lightDir;
    mat4 normalModel;
} ubuf;

layout(std140, binding = 2) uniform settings
{
    vec4 alpha; // cutoff, mode (0 opaque / 1 mask / 2 blend), factor.
} material;

layout(location = 0) out vec4 fragColor;

vec3 toLinear(vec3 c)
{
    return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92,
               lessThanEqual(c, vec3(0.04045)));
}

vec3 toSRGB(vec3 c)
{
    return mix(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92,
               lessThanEqual(c, vec3(0.0031308)));
}

void main()
{
    vec4 texel = texture(baseTexture, v_texcoord);
    float alpha = texel.a * material.alpha.z;
    int mode = int(material.alpha.y + 0.5);
    if (mode == 1 && alpha < material.alpha.x)
        discard;
    float light = 0.3 + 0.7 * abs(dot(normalize(v_normal), normalize(ubuf.lightDir.xyz)));
    vec3 color = toSRGB(max(toLinear(texel.rgb) * ubuf.color.rgb * light, vec3(0.0)));
    fragColor = vec4(color, (mode == 2 ? alpha : 1.0) * ubuf.color.a);
}
