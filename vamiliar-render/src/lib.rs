mod vamiliar_shader;
mod vamiliar_vertex;

use std::sync::Arc;
use vamiliar_shader::VamiliarShaders;
use vulkano::command_buffer::ClearColorImageError;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecError, CommandBufferUsage,
};
use vulkano::device::{Device, DeviceCreationError, DeviceExtensions, Features, Queue};
use vulkano::format::{ClearValue, Format};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreationError, PhysicalDevice, PhysicalDeviceType};
use vulkano::pipeline::input_assembly::PrimitiveTopology;
use vulkano::pipeline::vertex::BufferlessDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{
    GraphicsPipeline, GraphicsPipelineAbstract, GraphicsPipelineCreationError,
};
use vulkano::render_pass::{RenderPass, RenderPassCreationError, Subpass};
use vulkano::swapchain::{
    AcquireError, CapabilitiesError, ColorSpace, FullscreenExclusive, PresentMode, Surface,
    SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::{OomError, Version};
use vulkano_win::VkSurfaceBuild;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

#[derive(thiserror::Error, Debug)]
pub enum RenderInitError {
    #[error("failed to initialize Vulkan rendering API")]
    InstanceCreationError(#[from] InstanceCreationError),

    #[error("vulkan failed to locate any available physical devices")]
    NoPhysicalDevices,

    #[error("no available queue families on the chosen physical device")]
    NoAvailableQueueFamilies,

    #[error("failed to create device and queues")]
    DeviceCreationError(#[from] DeviceCreationError),

    #[error("failed to get a queue from the returned queues")]
    FailedToGetQueue,

    #[error("failed to create a surface onto which to render with Vulkan")]
    SurfaceCreationError(#[from] vulkano_win::CreationError),

    #[error("failed to create pipeline")]
    CreatePipelineError(#[from] CreatePipelineError),

    #[error("out of memory error")]
    OutOfMemoryError(#[from] OomError),
}

#[derive(thiserror::Error, Debug)]
pub enum CreateSwapchainError {
    #[error("failed to retrieve capabilities for swapchain")]
    CapabilitiesError(#[from] CapabilitiesError),

    #[error("no valid composite alpha mode found")]
    NoValidAlphaMode,

    #[error("no valid color space format found")]
    NoValidColorSpaceFormat,

    #[error("failed to create new swapchain")]
    SwapchainCreationError(#[from] SwapchainCreationError),
}

#[derive(thiserror::Error, Debug)]
pub enum RenderFrameError {
    #[error("failed to create a new swapchain")]
    CreateSwapchainError(#[from] CreateSwapchainError),

    #[error("failed to acquire new frame from vulkano")]
    AcquireError(#[from] AcquireError),

    #[error("failed to execute command buffer")]
    CommandBufferExecError(#[from] CommandBufferExecError),

    #[error("vulkan reported an out of memory error")]
    OutOfMemory(#[from] OomError),

    #[error("failed to clear color")]
    ClearColorImageError(#[from] ClearColorImageError),

    #[error("image index not found within swapchain")]
    InvalidImage,

    #[error("failed to build command buffer")]
    CommandBufferBuildError(#[from] vulkano::command_buffer::BuildError),

    #[error("failed to flush swapchain")]
    FlushError(#[from] FlushError),
}

#[derive(thiserror::Error, Debug)]
pub enum CreatePipelineError {
    #[error("failed to create graphics pipeline")]
    GraphicsPipelineCreationError(#[from] GraphicsPipelineCreationError),

    #[error("failed to create render pass")]
    RenderPassCreationError(#[from] RenderPassCreationError),

    #[error("failed to retrieve subpass {0}")]
    FailedToRetrieveSubpass(u32),
}

pub type SurfaceType = Arc<Surface<Window>>;
pub type SwapchainType = Arc<Swapchain<Window>>;
pub type SwapchainImageType = Arc<SwapchainImage<Window>>;

/// Hold information about the rendering context (until consumed to begin the event loop and start the game)
#[allow(dead_code)]
pub struct RenderContext {
    surface: SurfaceType,
    queue: Arc<Queue>,
    shaders: VamiliarShaders,
    graphics_pipeline: GraphicsPipeline<BufferlessDefinition>,
}

impl RenderContext {
    pub fn new() -> Result<(EventLoop<()>, RenderContext), RenderInitError> {
        // Get the instance extensions required to work with Winit
        let required_extensions = vulkano_win::required_extensions();

        // Create the Vulkan instance
        let instance = Instance::new(None, Version::V1_2, &required_extensions, None)?;

        // Create an event loop for Winit
        let event_loop = EventLoop::new();

        // Create a render surface for the Vulkan API to render onto
        let surface = WindowBuilder::new()
            .with_resizable(true)
            .with_title(concat!(
                // Default title
                env!("CARGO_PKG_NAME"),
                " v",
                env!("CARGO_PKG_VERSION")
            ))
            .build_vk_surface(&event_loop, instance.clone())?;

        // Decide which device extensions to load
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::none()
        };

        // Pick a specific physical device and queue family
        let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
            // Only load valid queue families for physical devices
            .filter_map(|p| {
                p.queue_families()
                    .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
                    .map(|q| (p, q))
            })
            // Pick the preferred physical device to use
            .min_by_key(|(p, _)| match p.properties().device_type {
                Some(PhysicalDeviceType::DiscreteGpu) => 0,
                Some(PhysicalDeviceType::IntegratedGpu) => 1,
                Some(PhysicalDeviceType::VirtualGpu) => 2,
                Some(PhysicalDeviceType::Cpu) => 3,
                Some(PhysicalDeviceType::Other) => 4,
                None => 5,
            })
            .ok_or(RenderInitError::FailedToGetQueue)?;

        // Create our link to the device and queues to which we can send commands
        let (device, mut queues) = Device::new(
            physical_device,
            &Features::none(),
            &DeviceExtensions::required_extensions(physical_device).union(&device_extensions),
            [(queue_family, 0.5)].iter().cloned(),
        )?;

        // Get a single queue (for now, I think?)
        let queue = queues.next().ok_or(RenderInitError::FailedToGetQueue)?;

        // Load the shaders
        let shaders = VamiliarShaders::load(device.clone())?;

        // Create the graphics pipeline
        let graphics_pipeline = Self::build_graphics_pipeline(&shaders, device)?;

        // Wrap & return the context
        let render_ctx = RenderContext {
            surface,
            queue,
            shaders,
            graphics_pipeline,
        };

        Ok((event_loop, render_ctx))
    }

    fn build_graphics_pipeline(
        shaders: &VamiliarShaders,
        device: Arc<Device>,
    ) -> Result<GraphicsPipeline<BufferlessDefinition>, CreatePipelineError> {
        // Create the render pass we will use (for now)
        let render_pass = Arc::new(Self::create_render_passes(device.clone())?);

        // Get the [first] subpass from the render pass
        let subpass = {
            let subpass_num = 0;
            Subpass::from(render_pass, subpass_num)
                .ok_or(CreatePipelineError::FailedToRetrieveSubpass(subpass_num))?
        };

        Ok(GraphicsPipeline::start()
            // Hide the back faces
            .cull_mode_back()
            // Clockwise points are considered fronts, counter-clockwise the
            // back. This is the *opposite* of OpenGL's default, but I find it
            // easier to think this way and it's my game engine, so I'll do as
            // I please. Thank you :]
            .front_face_clockwise()
            // I think this is like GL_TRIANGLES?
            .primitive_topology(PrimitiveTopology::TriangleList)
            // Fill texels in polygon
            .polygon_mode_fill()
            // Set the viewport (centered at 0.0)
            .viewports([Viewport {
                origin: [-1.0, -1.0],
                dimensions: [2.0, 2.0],
                depth_range: 0.0..1.0,
            }]) // Update the shaders to the precompiled versions
            .vertex_shader(
                shaders.vertex_shader.main_entry_point(),
                vamiliar_shader::vs::SpecializationConstants::default(),
            )
            .fragment_shader(
                shaders.fragment_shader.main_entry_point(),
                vamiliar_shader::fs::SpecializationConstants::default(),
            )
            // Set to use desired subpass
            .render_pass(subpass)
            // Build and yield the shiny new graphics pipeline
            .build(device)?)
    }

    #[allow(dead_code, clippy::blacklisted_name)]
    fn create_render_passes(device: Arc<Device>) -> Result<RenderPass, RenderPassCreationError> {
        vulkano::single_pass_renderpass!(device,
            attachments: {
                // `foo` is a custom name we give to the first and only attachment.
                foo: {
                    load: Clear,
                    store: Store,
                    format: Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [foo],       // Repeat the attachment name here.
                depth_stencil: {}
            }
        )
    }

    pub fn start_event_loop(&mut self, mut event_loop: EventLoop<()>) {
        // Take everything
        #[allow(unused_variables)]
        let (surface, device, queue) = (
            self.surface.clone(),
            self.queue.device().clone(),
            self.queue.clone(),
        );

        // Keep track of swapchain values (these will be moved into the closure
        // below as soon as the event loop begins)
        let mut swapchain = None;
        let mut swapchain_images = vec![];
        let mut swapchain_dirty = true;

        // Listen for events from Winit
        event_loop.run_return(move |event, _, control_flow| match &event {
            Event::WindowEvent {
                // Exit game when window closing
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,

            // Handle resizing
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                // Mark the swapchain as dirty
                swapchain_dirty = true;

                // Render a new frame (events are combined by winit)
                self.on_render(&mut swapchain, &mut swapchain_images, &mut swapchain_dirty)
                    .expect("failed to render frame");
            }

            // Render a new frame
            Event::MainEventsCleared => self
                .on_render(&mut swapchain, &mut swapchain_images, &mut swapchain_dirty)
                .expect("failed to render frame"),

            _ => {}
        });
    }

    fn on_render(
        &self,
        swapchain: &mut Option<SwapchainType>,
        swapchain_images: &mut Vec<SwapchainImageType>,
        swapchain_dirty: &mut bool,
    ) -> Result<(), RenderFrameError> {
        // If the swapchain hasn't been created yet, it should be marked as
        // dirty (assuming it hasn't been already).
        if swapchain.is_none() {
            *swapchain_dirty = true;
        }

        // Check if the swapchain needs to be recreated (window resize, etc)
        if *swapchain_dirty {
            // If the swapchain is dirty, we need to create a new one
            let (new_swapchain, new_swapchain_images) = self.create_swapchain(swapchain)?;

            // Update the global values
            *swapchain = Some(new_swapchain);
            *swapchain_images = new_swapchain_images;

            // Mark the swapchain as no longer dirty
            *swapchain_dirty = false;
        }

        // Render a frame
        self.render_frame(
            swapchain.as_ref().unwrap().clone(),
            swapchain_images,
            swapchain_dirty,
        )
        .expect("failed to render frame");

        Ok(())
    }

    fn create_swapchain(
        &self,
        current_swapchain: &Option<SwapchainType>,
    ) -> Result<(SwapchainType, Vec<SwapchainImageType>), CreateSwapchainError> {
        let surface = self.surface.clone();
        let device = self.queue.device().clone();

        // Get the capabilities for a swapchain
        let capabilities = surface.capabilities(device.physical_device())?;

        let [w, h] = capabilities.current_extent.unwrap_or([1280, 720]);
        println!("creating new swapchain of size {}x{}", w, h);

        let new_sw = if let Some(current_swapchain) = current_swapchain {
            // Build from the current swapchain
            current_swapchain.recreate()
        } else {
            // Get basic information from the capabilities
            let alpha = capabilities
                .supported_composite_alpha
                .iter()
                .next()
                .ok_or(CreateSwapchainError::NoValidAlphaMode)?;
            let (format, _) = capabilities
                .supported_formats
                .get(0)
                .ok_or(CreateSwapchainError::NoValidColorSpaceFormat)?;
            // Try to use double-buffering.
            let buffers_count = match capabilities.max_image_count {
                None => 2.max(capabilities.min_image_count),
                Some(limit) => limit.min(2.max(capabilities.min_image_count)),
            };

            // Create a new swapchain from scratch
            Swapchain::start(device, surface)
                .num_images(buffers_count)
                .usage(ImageUsage::color_attachment())
                .transform(SurfaceTransform::Identity)
                .composite_alpha(alpha)
                .format(*format)
                .present_mode(PresentMode::Fifo)
                .fullscreen_exclusive(FullscreenExclusive::Allowed)
                .color_space(ColorSpace::SrgbNonLinear)
        };

        // Create the new swapchain
        Ok(new_sw.dimensions([w, h]).build()?)
    }

    fn render_frame(
        &self,
        swapchain: SwapchainType,
        swapchain_images: &[SwapchainImageType],
        swapchain_dirty: &mut bool,
    ) -> Result<(), RenderFrameError> {
        // Acquire the next image to render from vulkan (no timeout for now)
        let (image_num, suboptimal, acquire_future) =
            vulkano::swapchain::acquire_next_image(swapchain.clone(), None)?;

        // If the swapchain is suboptimal, mark it as needing recreated
        if suboptimal {
            *swapchain_dirty = true;
            println!("image acquisition was suboptimal, the swapchain should be recreated");
        }

        // Get the specified image from the swapchain
        let image = swapchain_images
            .get(image_num)
            .ok_or(RenderFrameError::InvalidImage)?;

        // Create the primary command buffer
        let mut primary_command_buffer = AutoCommandBufferBuilder::primary(
            self.queue.device().clone(),
            self.queue.family(),
            CommandBufferUsage::MultipleSubmit,
        )?;
        primary_command_buffer
            // Add a clear command first
            .clear_color_image(image.clone(), ClearValue::Float([0.2, 0.4, 0.6, 1.0]))?;
        let pcb = primary_command_buffer.build()?;

        // Execute the command buffer and display the result to the screen
        Ok(acquire_future
            .then_execute(self.queue.clone(), pcb)?
            .then_swapchain_present(self.queue.clone(), swapchain, image_num)
            .then_signal_fence_and_flush()
            .map(|_| ())?)
    }

    pub fn window(&self) -> &Window {
        self.surface.window()
    }
}
