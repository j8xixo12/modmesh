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
    vec4 surface; // roughness, metallic, reserved, studio enabled.
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


vec3 studioLight(vec3 base, vec3 normal, vec3 view)
{
    vec3 n = normalize(normal);
    vec3 v = normalize(view);
    if (dot(n, v) < 0.0) n = -n;
    vec3 key = normalize(v + vec3(-0.6, -0.4, 0.9));
    vec3 fill = normalize(v + vec3(0.8, 0.5, 0.2));
    vec3 rim = normalize(-v + vec3(0.0, 0.0, 1.2));
    float roughness = clamp(ubuf.surface.x, 0.06, 1.0);
    float metal = ubuf.surface.y;
    float power = mix(180.0, 4.0, roughness * roughness);
    vec3 halfDir = normalize(key + v);
    vec3 f0 = mix(vec3(0.04), base, metal);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, halfDir), 0.0), 5.0);
    vec3 diffuse = base * (1.0 - metal * 0.75)
        * (0.20 + 0.70 * max(dot(n, key), 0.0)
            + 0.25 * max(dot(n, fill), 0.0));
    vec3 specular = fresnel * pow(max(dot(n, halfDir), 0.0), power)
        * (1.0 - roughness * 0.65) * 1.8;
    vec3 edge = vec3(0.65, 0.76, 1.0) * max(dot(n, rim), 0.0)
        * (0.08 + metal * 0.18);
    return max(diffuse + specular + edge, vec3(0.0));
}

void main()
{
    vec4 texel = texture(baseTexture, v_texcoord);
    float alpha = texel.a * material.alpha.z;
    int mode = int(material.alpha.y + 0.5);
    if (mode == 1 && alpha < material.alpha.x)
        discard;
    float light = 0.3 + 0.7 * abs(dot(normalize(v_normal), normalize(ubuf.lightDir.xyz)));
    vec3 base = toLinear(texel.rgb) * ubuf.color.rgb;
    vec3 shaded = ubuf.surface.w > 0.5
        ? studioLight(base, v_normal, ubuf.lightDir.xyz) : base * light;
    vec3 color = toSRGB(max(shaded, vec3(0.0)));
    fragColor = vec4(color, (mode == 2 ? alpha : 1.0) * ubuf.color.a);
}
