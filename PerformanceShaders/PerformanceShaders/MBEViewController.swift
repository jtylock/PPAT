import UIKit
import Metal
import MetalKit
import MetalPerformanceShaders

// Size of buffers that hold vertex and uniform data.
// These bounds are pretty tight, so if you modify this
// code to draw more geometry, you'll want to make them larger.
let MBEVertexDataSize = 128
let MBEUniformDataSize = 64
let MBEMaxInflightBuffers = 3

// Vertex data for drawing a textured square
// The texture coordinates (s, t) below simply select an
// interesting square region of the included texture.
var vertexData:[Float32] =
[
//    x     y    z    w    s    t
    -1.5,  1.5, 0.0, 1.0, 0.0, 0.0,
    -1.5, -1.5, 0.0, 1.0, 0.0, 1.0,
     1.5,  1.5, 0.0, 1.0, 1.0, 0.0,
     1.5, -1.5, 0.0, 1.0, 1.0, 1.0,
]

// This copying allocator can be used by certain Metal Performance Shader kernels
// to allocate a target texture when the are unable to operate in-place
let MBEFallbackAllocator =
{ (kernel: MPSKernel, commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture) -> MTLTexture in
    return sourceTexture.device.newTextureWithDescriptor(sourceTexture.matchingDescriptor())
}

class MBEViewController:UIViewController, MTKViewDelegate, UIImagePickerControllerDelegate,UINavigationControllerDelegate  {
    
    let device: MTLDevice = MTLCreateSystemDefaultDevice()!
    
    var commandQueue: MTLCommandQueue! = nil
    var pipelineState: MTLRenderPipelineState! = nil
    var vertexBuffer: MTLBuffer! = nil
    var uniformBuffer: MTLBuffer! = nil
    var sampler: MTLSamplerState! = nil

    var kernelSourceTexture: MTLTexture? = nil
    var kernelDestTexture: MTLTexture? = nil

    let inflightSemaphore = dispatch_semaphore_create(MBEMaxInflightBuffers)
    var bufferIndex = 0

    var gaussianBlurKernel: MPSUnaryImageKernel!
    var thresholdKernel: MPSUnaryImageKernel!
    var edgeKernel: MPSUnaryImageKernel!
    var saturationKernel: MBEImageSaturation!
    var filterImage:UIImage!
    var selectedKernel: MPSUnaryImageKernel!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        assert(MPSSupportsMTLDevice(device), "This device does not support Metal Performance Shaders")
        let view = self.view as! MTKView
        view.device = device
        view.delegate = self
        filterImage = UIImage(named: "mandrill")
        buildKernels()
        loadAssets()
    }

    override func prefersStatusBarHidden() -> Bool {
        return true
    }

    func buildKernels() {
        gaussianBlurKernel = MPSImageLaplacian(device: device)
        thresholdKernel = MPSImageThresholdToZero(device: device, thresholdValue: 0.5, linearGrayColorTransform: nil)
        edgeKernel = MPSImageSobel(device: device)
        saturationKernel = MBEImageSaturation(device: device, saturationFactor: 0)
        selectedKernel = saturationKernel
    }

    func loadAssets() {
        let view = self.view as! MTKView
        commandQueue = device.newCommandQueue()
        commandQueue.label = "Command queue"
        let defaultLibrary = device.newDefaultLibrary()!
        let vertexProgram = defaultLibrary.newFunctionWithName("project_vertex")!
        let fragmentProgram = defaultLibrary.newFunctionWithName("texture_fragment")!
        let pipelineStateDescriptor = MTLRenderPipelineDescriptor()
        pipelineStateDescriptor.vertexDescriptor = MBECreateVertexDescriptor()
        pipelineStateDescriptor.vertexFunction = vertexProgram
        pipelineStateDescriptor.fragmentFunction = fragmentProgram
        pipelineStateDescriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pipelineStateDescriptor.sampleCount = view.sampleCount

        let samplerDescriptor = MTLSamplerDescriptor()
        samplerDescriptor.minFilter = .Nearest
        samplerDescriptor.magFilter = .Linear
        sampler = device.newSamplerStateWithDescriptor(samplerDescriptor)

        let textureLoader = MTKTextureLoader(device: device)
       
        do {
            if let image = filterImage.flippedImage().flippedImage().CGImage, let image2 = filterImage.flippedImage().CGImage{
                try kernelSourceTexture = textureLoader.newTextureWithCGImage(image, options: nil)
                kernelDestTexture = device.newTextureWithDescriptor(kernelSourceTexture!.matchingDescriptor())
            } else {
                print("Failed to load texture image from main bundle")
            }
        }
        catch let error {
            print("Failed to create texture from image, error \(error)")
        }

        do {
            try pipelineState = device.newRenderPipelineStateWithDescriptor(pipelineStateDescriptor)
        } catch let error {
            print("Failed to create pipeline state, error \(error)")
        }
        

        vertexBuffer = device.newBufferWithLength(MBEVertexDataSize * MBEMaxInflightBuffers, options: [])
        vertexBuffer.label = "vertices"
        
        uniformBuffer = device.newBufferWithLength(MBEUniformDataSize * MBEMaxInflightBuffers, options: [])
        uniformBuffer.label = "uniforms"
    }

    @IBOutlet weak var ImageView: UIImageView!
    
    
    @IBAction func cameraAction(sender: UIButton) {
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerControllerSourceType.Camera){
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = UIImagePickerControllerSourceType.Camera;
            imagePicker.allowsEditing = false
            self.presentViewController(imagePicker, animated: true, completion: nil)
            
        }
    }
    
    @IBAction func photolibraryaction(sender: UIButton) {
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerControllerSourceType.PhotoLibrary){
            let imagePicker = UIImagePickerController()
            imagePicker.delegate = self
            imagePicker.sourceType = UIImagePickerControllerSourceType.PhotoLibrary;
            imagePicker.allowsEditing = false
        }

    }
    
    func imagePickerController( picker:UIImagePickerController, didFinishPickingImage image: UIImage!,editingInfo: [NSObject: AnyObject]!){
        filterImage = image
        buildKernels()
        loadAssets()
        self.dismissViewControllerAnimated(true, completion:nil);
    }
    
    @IBAction func selectedKernelChanged(sender: UISegmentedControl) {
        switch sender.selectedSegmentIndex {
        case 0: selectedKernel = gaussianBlurKernel
        case 1: selectedKernel = thresholdKernel
        case 2: selectedKernel = edgeKernel
        case 3: selectedKernel = saturationKernel
        default: break
        }
    }

    func updateBuffers() {
        // Copy positions and tex coords to the portion of the vertex buffer we'll be rendering from
        let vertBufferPtr = vertexBuffer.contents()
        let currentVertPtr = UnsafeMutablePointer<Float>(vertBufferPtr + MBEVertexDataSize * bufferIndex)
        memcpy(currentVertPtr, vertexData, 24 * sizeof(Float32))

        // Build the uniforms for the current frame. If we were animating, we could do it here
        let aspect = Float32(self.view.bounds.width / self.view.bounds.height)
        let fov = Float32(M_PI / 2)
        let projectionMatrix =  matrix_perspective_projection(aspect, fieldOfViewYRadians: fov, near: 0.1, far: 10.0)
        let viewMatrix = matrix_translation([0, 0, -2])
        var uniforms = MBEUniforms(modelViewProjectionMatrix: projectionMatrix * viewMatrix)

        // Copy uniform data into the portion of the uniform buffer we'll be rendering from
        let uniformBufferPtr = uniformBuffer.contents()
        let currentUniformPtr = UnsafeMutablePointer<Float>(uniformBufferPtr + MBEUniformDataSize * bufferIndex)
        memcpy(currentUniformPtr, &uniforms, sizeof(MBEUniforms))
    }
    
    func drawInMTKView(view: MTKView) {
        
        dispatch_semaphore_wait(inflightSemaphore, DISPATCH_TIME_FOREVER)
        updateBuffers()
        let commandBuffer = commandQueue.commandBuffer()
        commandBuffer.addCompletedHandler{ [weak self] commandBuffer in
            if let strongSelf = self {
                dispatch_semaphore_signal(strongSelf.inflightSemaphore)
            }
        }

        // Update the saturation kernel's time-varying saturation factor
        saturationKernel.saturationFactor = Float32(abs(sin(CACurrentMediaTime() * 2)))

        // Dispatch the current kernel to perform the selected image filter
        selectedKernel.encode(commandBuffer: commandBuffer,
            sourceTexture: kernelSourceTexture!,
            destinationTexture: kernelDestTexture!)

        if let renderPassDescriptor = view.currentRenderPassDescriptor, let currentDrawable = view.currentDrawable
        {
            let clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
            renderPassDescriptor.colorAttachments[0].clearColor = clearColor
            let renderEncoder = commandBuffer.renderCommandEncoderWithDescriptor(renderPassDescriptor)
            renderEncoder.label = "Main pass"
            renderEncoder.pushDebugGroup("Draw textured square")
            renderEncoder.setFrontFacingWinding(.CounterClockwise)
            renderEncoder.setCullMode(.Back)
            renderEncoder.setRenderPipelineState(pipelineState)
            renderEncoder.setVertexBuffer(vertexBuffer, offset: MBEVertexDataSize * bufferIndex, atIndex: 0)
            renderEncoder.setVertexBuffer(uniformBuffer, offset: MBEUniformDataSize * bufferIndex , atIndex: 1)
            renderEncoder.setFragmentTexture(kernelDestTexture, atIndex: 0)
            renderEncoder.setFragmentSamplerState(sampler, atIndex: 0)
            renderEncoder.drawPrimitives(.TriangleStrip, vertexStart: 0, vertexCount: 4)
            renderEncoder.popDebugGroup()
            renderEncoder.endEncoding()
            commandBuffer.presentDrawable(currentDrawable)
        }
        
        bufferIndex = (bufferIndex + 1) % MBEMaxInflightBuffers
        
        commandBuffer.commit()
    }

    func mtkView(view: MTKView, drawableSizeWillChange size: CGSize) {
    }
}
