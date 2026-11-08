#version 440

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    vec4 color;
    vec4 lightDir;
    mat4 normalModel;
} ubuf;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 v_texcoord;

void main()
{
    v_normal = mat3(ubuf.normalModel) * normal;
    v_texcoord = texcoord;
    gl_Position = ubuf.mvp * vec4(position, 1.0);
}
