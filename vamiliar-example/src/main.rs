use vamiliar::vamiliar_render::RenderContext;

fn main() {
    println!("initializing");

    // Initialize the render context
    let (event_loop, mut render_context) =
        RenderContext::new().expect("failed to create render context");
    render_context.window().set_title(concat!(
        // Default title
        env!("CARGO_PKG_NAME"),
        " v",
        env!("CARGO_PKG_VERSION")
    ));

    println!("starting event loop");

    // Start the event loop
    render_context.start_event_loop(event_loop);

    println!("exiting");
}
