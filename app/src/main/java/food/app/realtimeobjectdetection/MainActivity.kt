package food.app.realtimeobjectdetection

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.*
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import android.view.TextureView
import android.widget.Toast
import androidx.core.content.ContextCompat
import food.app.realtimeobjectdetection.databinding.ActivityMainBinding
import food.app.realtimeobjectdetection.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    private lateinit var cameraManager: CameraManager
    private lateinit var handler: Handler
    private lateinit var cameraDevice: CameraDevice
    private lateinit var bitmap: Bitmap
    private lateinit var model: SsdMobilenetV11Metadata1
    private lateinit var imageProcess: ImageProcessor

    private lateinit var labels: List<String>

    val paint = Paint()

    var colors = listOf(
        Color.BLUE,
        Color.GREEN,
        Color.RED,
        Color.CYAN,
        Color.GRAY,
        Color.GRAY,
        Color.BLACK,
        Color.DKGRAY,
        Color.MAGENTA,
        Color.YELLOW,
        Color.RED
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // get Users Permission
        getPermission()

        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        imageProcess =
            ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()

        model = SsdMobilenetV11Metadata1.newInstance(applicationContext)

        val handlerThread = HandlerThread("videoThread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        labels = FileUtil.loadLabels(this, "labels.txt")

        binding.TextureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {

            override fun onSurfaceTextureAvailable(
                surface: SurfaceTexture,
                width: Int,
                height: Int,
            ) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(
                surface: SurfaceTexture,
                width: Int,
                height: Int,
            ) {
                Log.e("myCamera", "Surface Texture Size Changed ")
            }

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                bitmap = binding.TextureView.bitmap!!

                // Creates inputs for reference.
                var image = TensorImage.fromBitmap(bitmap)

                image = imageProcess.process(image)

                // Runs model inference and gets result.
                val outputs = model.process(image)
                val locations = outputs.locationsAsTensorBuffer.floatArray
                val classes = outputs.classesAsTensorBuffer.floatArray
                val scores = outputs.scoresAsTensorBuffer.floatArray
                val numberOfDetections = outputs.numberOfDetectionsAsTensorBuffer.floatArray

                val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutable)

                val h = mutable.height
                val w = mutable.width

                paint.textSize = h / 15f
                paint.strokeWidth = h / 85f

                var x = 0

                scores.forEachIndexed { index, fl ->

                    x = index
                    x *= 4

                    if (fl > 0.5) {

                        paint.color = colors[index]
                        paint.style = Paint.Style.STROKE
                        canvas.drawRect(
                            RectF(
                                locations[x + 1] * w,
                                locations[x] * h,
                                locations[x + 3] * w,
                                locations[x + 2] * h
                            ), paint
                        )
                        paint.style = Paint.Style.FILL
                        canvas.drawText(
                            labels[classes[index].toInt()] + " " + fl.toString(),
                            locations[x + 1] * w, locations[x] * h, paint
                        )

                    }

                }

                // get the bitmap to screen
                binding.imageView.setImageBitmap(mutable)


            }

        }

    }

    @SuppressLint("MissingPermission")
    private fun openCamera() {


        cameraManager.openCamera(
            cameraManager.cameraIdList[0],
            object : CameraDevice.StateCallback() {

                override fun onOpened(camera: CameraDevice) {

                    cameraDevice = camera

                    val surfaceTexture = binding.TextureView.surfaceTexture

                    val surface = Surface(surfaceTexture)

                    val captureRequest =
                        cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)

                    captureRequest.addTarget(surface)

                    cameraDevice.createCaptureSession(
                        listOf(surface),
                        object : CameraCaptureSession.StateCallback() {

                            override fun onConfigured(session: CameraCaptureSession) {

                                session.setRepeatingRequest(captureRequest.build(), null, null)

                            }

                            override fun onConfigureFailed(session: CameraCaptureSession) {
                                Log.e("myCamera", " camera Configuration Fail ")
                            }

                        },
                        handler
                    )


                }

                override fun onDisconnected(camera: CameraDevice) {
                    Toast.makeText(this@MainActivity, "Camera disconnected ", Toast.LENGTH_LONG)
                        .show()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    Toast.makeText(this@MainActivity, "error occurred: $error ", Toast.LENGTH_LONG)
                        .show()

                }

            },
            handler
        )

    }

    override fun onDestroy() {
        super.onDestroy()
        // Releases model resources if no longer used.
        model.close()
    }


    // code for getting User Permissions for Camera
    private fun getPermission() {

        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(android.Manifest.permission.CAMERA), 101)
        }

    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getPermission()
        }
    }

}