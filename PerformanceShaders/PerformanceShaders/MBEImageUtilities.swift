import UIKit

extension UIImage {
    /// Utility function for flipping this image around the horizontal axis
    func flippedImage() -> UIImage {
        let imageSize = self.size
        UIGraphicsBeginImageContextWithOptions(imageSize, false, self.scale)
        let context = UIGraphicsGetCurrentContext()
        CGContextTranslateCTM(context!, 0, imageSize.height)
        CGContextScaleCTM(context!, 1, -1)
        self.drawInRect(CGRectMake(0, 0, imageSize.width, imageSize.height))
        let flippedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return flippedImage!
    }
}
