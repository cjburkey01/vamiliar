#[derive(Default, Copy, Clone)]
pub struct Vertex {
    position: [f32; 3],
}

vulkano::impl_vertex!(Vertex, position);
