use std::sync::Arc;
use vulkano::device::Device;
use vulkano::OomError;

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            layout(location = 0) in vec3 position;

            void main() {
                gl_Position = vec4(position, 1.0);
            }
        "
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) out vec4 fragColor;

            void main() {
                fragColor = vec4(1.0, 1.0, 1.0, 1.0);
            }
        "
    }
}

pub struct VamiliarShaders {
    pub vertex_shader: vs::Shader,
    pub fragment_shader: fs::Shader,
}

impl VamiliarShaders {
    pub fn load(device: Arc<Device>) -> Result<Self, OomError> {
        Ok(Self {
            vertex_shader: vs::Shader::load(device.clone())?,
            fragment_shader: fs::Shader::load(device)?,
        })
    }
}
