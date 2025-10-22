// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//
//  This file is part of the Ultralytics YOLO Package, providing the main entry point for using YOLO models.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The YOLO class serves as the primary interface for loading and using YOLO machine learning models.
//  It supports a variety of input formats including UIImage, CIImage, CGImage, and resource files.
//  The class handles model loading, format conversion, and inference execution, offering a simple yet
//  powerful API through Swift's callable object pattern.
//
//  Model Loading Priority:
//  1. Absolute file paths (e.g., /Users/username/models/yolo.mlpackage)
//  2. Relative paths with model extensions (e.g., models/yolo.mlpackage)
//  3. Bundle resources by name (e.g., "yolo11n" searches for yolo11n.mlpackage in app bundle)
//  4. Flutter assets (for backward compatibility with Flutter apps)
//
//  Supports loading models from anywhere in the file system, not just the app bundle.

import Foundation
import SwiftUI
import UIKit

/// The primary interface for working with YOLO models, supporting multiple input types and inference methods.
///
/// Example usage:
/// ```swift
/// // Load from app bundle
/// let yolo1 = YOLO("yolo11n", task: .detect) { result in
///   // handle result
/// }
///
/// // Load from absolute path (e.g., Documents directory)
/// let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
/// let modelPath = documentsPath.appendingPathComponent("my_model.mlpackage").path
/// let yolo2 = YOLO(modelPath, task: .detect) { result in
///   // handle result
/// }
///
/// // Load from any file system location
/// let yolo3 = YOLO("/Users/username/Downloads/custom_model.mlpackage", task: .segment) { result in
///   // handle result
/// }
/// ```
public class YOLO {
  var predictor: Predictor!

  /// Confidence threshold for filtering predictions (0.0-1.0)
  public var confidenceThreshold: Double = 0.25 {
    didSet {
      // Apply to predictor if it has been loaded
      if let basePredictor = predictor as? BasePredictor {
        basePredictor.setConfidenceThreshold(confidence: confidenceThreshold)
      }
    }
  }

  /// IoU threshold for non-maximum suppression (0.0-1.0)
  public var iouThreshold: Double = 0.4 {
    didSet {
      // Apply to predictor if it has been loaded
      if let basePredictor = predictor as? BasePredictor {
        basePredictor.setIouThreshold(iou: iouThreshold)
      }
    }
  }

  public init(
    _ modelPathOrName: String, task: YOLOTask, useGpu: Bool = true,
    completion: ((Result<YOLO, Error>) -> Void)? = nil
  ) {
    print("YOLO.init: Received modelPath: \(modelPathOrName)")

    var modelURL: URL?

    let lowercasedPath = modelPathOrName.lowercased()
    let fileManager = FileManager.default

    // Step 1: Check if it's an absolute file path (anywhere in the file system)
    // This includes paths outside the app bundle like Documents, Downloads, etc.
    if modelPathOrName.hasPrefix("/") || modelPathOrName.hasPrefix("file://") {
      let absolutePath = modelPathOrName.hasPrefix("file://")
        ? URL(string: modelPathOrName)?.path ?? modelPathOrName
        : modelPathOrName

      var isDirectory: ObjCBool = false
      if fileManager.fileExists(atPath: absolutePath, isDirectory: &isDirectory) {
        let url = URL(fileURLWithPath: absolutePath)

        // Validate the file extension matches the expected type
        if lowercasedPath.hasSuffix(".mlpackage") && isDirectory.boolValue {
          modelURL = url
          print("YOLO.init: Found .mlpackage directory at absolute path: \(absolutePath)")
        } else if lowercasedPath.hasSuffix(".mlmodel") && !isDirectory.boolValue {
          modelURL = url
          print("YOLO.init: Found .mlmodel file at absolute path: \(absolutePath)")
        } else if lowercasedPath.hasSuffix(".mlmodelc") && isDirectory.boolValue {
          modelURL = url
          print("YOLO.init: Found .mlmodelc directory at absolute path: \(absolutePath)")
        } else {
          print("YOLO.init: File exists at \(absolutePath) but doesn't match expected type (isDirectory: \(isDirectory.boolValue))")
        }
      } else {
        print("YOLO.init: Absolute path does not exist: \(absolutePath)")
      }
    }

    // Step 2: Check if it has a model extension but isn't an absolute path
    // (could be a relative path that should be treated as absolute)
    if modelURL == nil && (lowercasedPath.hasSuffix(".mlmodel") || lowercasedPath.hasSuffix(".mlpackage") || lowercasedPath.hasSuffix(".mlmodelc")) {
      let possibleURL = URL(fileURLWithPath: modelPathOrName)
      var isDirectory: ObjCBool = false
      if fileManager.fileExists(atPath: possibleURL.path, isDirectory: &isDirectory) {
        if lowercasedPath.hasSuffix(".mlpackage") && isDirectory.boolValue {
          modelURL = possibleURL
          print("YOLO.init: Found .mlpackage directory at relative path: \(possibleURL.path)")
        } else if lowercasedPath.hasSuffix(".mlmodel") && !isDirectory.boolValue {
          modelURL = possibleURL
          print("YOLO.init: Found .mlmodel file at relative path: \(possibleURL.path)")
        } else if lowercasedPath.hasSuffix(".mlmodelc") && isDirectory.boolValue {
          modelURL = possibleURL
          print("YOLO.init: Found .mlmodelc directory at relative path: \(possibleURL.path)")
        }
      }
    }

    // Step 3: Check in app bundle by resource name (no extension in path)
    if modelURL == nil && !lowercasedPath.contains("/") {
      if let compiledURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlmodelc") {
        modelURL = compiledURL
        print("YOLO.init: Found .mlmodelc in bundle: \(modelPathOrName)")
      } else if let packageURL = Bundle.main.url(forResource: modelPathOrName, withExtension: "mlpackage") {
        modelURL = packageURL
        print("YOLO.init: Found .mlpackage in bundle: \(modelPathOrName)")
      }
    }

    // Step 4: Check for Flutter assets (for backward compatibility)
    if modelURL == nil {
      print("YOLO.init: Checking Flutter assets for: \(modelPathOrName)")

      // フォルダ構造を持つパスの場合
      if modelPathOrName.contains("/") && modelURL == nil {
        let components = modelPathOrName.components(separatedBy: "/")
        let fileName = components.last ?? ""
        let directory = components.dropLast().joined(separator: "/")
        let assetDirectory = "flutter_assets/\(directory)"

        print("YOLO Debug: Checking in asset directory: \(assetDirectory) for file: \(fileName)")

        // ファイル名をそのまま使用
        if let assetPath = Bundle.main.path(
          forResource: fileName, ofType: nil, inDirectory: assetDirectory)
        {
          print("YOLO Debug: Found model in assets directory: \(assetPath)")
          modelURL = URL(fileURLWithPath: assetPath)
        }

        // 拡張子を分割して検索
        if modelURL == nil && fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          print(
            "YOLO Debug: Trying with separated name: \(name) and extension: \(ext) in directory: \(assetDirectory)"
          )

          if let assetPath = Bundle.main.path(
            forResource: name, ofType: ext, inDirectory: assetDirectory)
          {
            print("YOLO Debug: Found model with separated extension: \(assetPath)")
            modelURL = URL(fileURLWithPath: assetPath)
          }
        }
      }

      // アセットディレクトリを直接確認
      if modelURL == nil && modelPathOrName.contains("/") {
        let assetPath = "flutter_assets/\(modelPathOrName)"
        print("YOLO Debug: Checking direct asset path: \(assetPath)")

        if let directPath = Bundle.main.path(forResource: assetPath, ofType: nil) {
          print("YOLO Debug: Found model at direct asset path: \(directPath)")
          modelURL = URL(fileURLWithPath: directPath)
        }
      }

      // フォルダ構造がない場合は、ファイル名だけで検索
      if modelURL == nil {
        let fileName = modelPathOrName.components(separatedBy: "/").last ?? modelPathOrName
        print("YOLO Debug: Checking filename only: \(fileName) in flutter_assets root")

        // Flutterアセットルートをチェック
        if let assetPath = Bundle.main.path(
          forResource: fileName, ofType: nil, inDirectory: "flutter_assets")
        {
          print("YOLO Debug: Found model in flutter_assets root: \(assetPath)")
          modelURL = URL(fileURLWithPath: assetPath)
        }

        // 拡張子を分割して検索
        if modelURL == nil && fileName.contains(".") {
          let fileComponents = fileName.components(separatedBy: ".")
          let name = fileComponents.dropLast().joined(separator: ".")
          let ext = fileComponents.last ?? ""

          print("YOLO Debug: Trying with separated filename: \(name) and extension: \(ext)")

          if let assetPath = Bundle.main.path(
            forResource: name, ofType: ext, inDirectory: "flutter_assets")
          {
            print(
              "YOLO Debug: Found model with separated extension in flutter_assets: \(assetPath)")
            modelURL = URL(fileURLWithPath: assetPath)
          }
        }
      }
    }

    // リソースバンドル内での確認 (例：Example/Flutter/App.frameworks/App.framework)
    if modelURL == nil {
      for bundle in Bundle.allBundles {
        let bundleID = bundle.bundleIdentifier ?? "unknown"
        print("YOLO Debug: Checking bundle: \(bundleID)")

        // フォルダ構造がある場合
        if modelPathOrName.contains("/") {
          let components = modelPathOrName.components(separatedBy: "/")
          let fileName = components.last ?? ""

          // ファイル名のみを検索
          if let path = bundle.path(forResource: fileName, ofType: nil) {
            print("YOLO Debug: Found model in bundle \(bundleID): \(path)")
            modelURL = URL(fileURLWithPath: path)
            break
          }

          // 拡張子を分割して検索
          if fileName.contains(".") {
            let fileComponents = fileName.components(separatedBy: ".")
            let name = fileComponents.dropLast().joined(separator: ".")
            let ext = fileComponents.last ?? ""

            if let path = bundle.path(forResource: name, ofType: ext) {
              print("YOLO Debug: Found model with ext in bundle \(bundleID): \(path)")
              modelURL = URL(fileURLWithPath: path)
              break
            }
          }
        }
      }
    }

    guard let unwrappedModelURL = modelURL else {
      print("YOLO Error: Model not found at path: \(modelPathOrName)")
      print("YOLO Debug: Original model path: \(modelPathOrName)")
      print("YOLO Debug: Lowercased path: \(lowercasedPath)")

      // Check if the path exists as directory
      var isDirectory: ObjCBool = false
      if fileManager.fileExists(atPath: modelPathOrName, isDirectory: &isDirectory) {
        print("YOLO Debug: Path exists. Is directory: \(isDirectory.boolValue)")

        // If it's a directory and ends with .mlpackage, it should have been found
        if isDirectory.boolValue && lowercasedPath.hasSuffix(".mlpackage") {
          print("YOLO Error: mlpackage directory exists but was not properly recognized")
        }
      } else {
        print("YOLO Debug: Path does not exist")
      }

      // 利用可能なバンドルと資産の一覧を表示
      print("YOLO Debug: Available bundles:")
      for bundle in Bundle.allBundles {
        print(" - \(bundle.bundleIdentifier ?? "unknown"): \(bundle.bundlePath)")
      }
      print("YOLO Debug: Checking if flutter_assets directory exists:")
      let flutterAssetsPath = Bundle.main.bundlePath + "/flutter_assets"
      if fileManager.fileExists(atPath: flutterAssetsPath) {
        print(" - flutter_assets exists at: \(flutterAssetsPath)")
        // flutter_assets内のファイル一覧を取得
        do {
          let items = try fileManager.contentsOfDirectory(atPath: flutterAssetsPath)
          print("YOLO Debug: Files in flutter_assets:")
          for item in items {
            print(" - \(item)")
          }
        } catch {
          print("YOLO Debug: Error listing flutter_assets: \(error)")
        }
      } else {
        print(" - flutter_assets NOT found")
      }

      completion?(.failure(PredictorError.modelFileNotFound))
      return
    }

    func handleSuccess(predictor: Predictor) {
      self.predictor = predictor
      completion?(.success(self))
    }

    // Common failure handling for all tasks
    func handleFailure(_ error: Error) {
      print("Failed to load model with error: \(error)")
      completion?(.failure(error))
    }

    switch task {
    case .classify:
      Classifier.create(unwrappedModelURL: unwrappedModelURL, useGpu: useGpu) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .segment:
      Segmenter.create(unwrappedModelURL: unwrappedModelURL, useGpu: useGpu) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .pose:
      PoseEstimater.create(unwrappedModelURL: unwrappedModelURL, useGpu: useGpu) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    case .obb:
      ObbDetector.create(unwrappedModelURL: unwrappedModelURL, useGpu: useGpu) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }

    default:
      ObjectDetector.create(unwrappedModelURL: unwrappedModelURL, useGpu: useGpu) { result in
        switch result {
        case .success(let predictor):
          handleSuccess(predictor: predictor)
        case .failure(let error):
          handleFailure(error)
        }
      }
    }
  }

  public func callAsFunction(_ uiImage: UIImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    let ciImage = CIImage(image: uiImage)!
    var result = predictor.predictOnImage(image: ciImage)
    //        if returnAnnotatedImage {
    //            let annotatedImage = drawYOLODetections(on: ciImage, result: result)
    //            result.annotatedImage = annotatedImage
    //        }
    return result
  }

  public func callAsFunction(_ ciImage: CIImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    var result = predictor.predictOnImage(image: ciImage)
    if returnAnnotatedImage {
      let annotatedImage = drawYOLODetections(on: ciImage, result: result)
      result.annotatedImage = annotatedImage
    }
    return result
  }

  public func callAsFunction(_ cgImage: CGImage, returnAnnotatedImage: Bool = true) -> YOLOResult {
    let ciImage = CIImage(cgImage: cgImage)
    var result = predictor.predictOnImage(image: ciImage)
    if returnAnnotatedImage {
      let annotatedImage = drawYOLODetections(on: ciImage, result: result)
      result.annotatedImage = annotatedImage
    }
    return result
  }

  public func callAsFunction(
    _ resourceName: String,
    withExtension ext: String? = nil,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    guard let url = Bundle.main.url(forResource: resourceName, withExtension: ext),
      let data = try? Data(contentsOf: url),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  public func callAsFunction(
    _ remoteURL: URL?,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    guard let remoteURL = remoteURL,
      let data = try? Data(contentsOf: remoteURL),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  public func callAsFunction(
    _ localPath: String,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    let fileURL = URL(fileURLWithPath: localPath)
    guard let data = try? Data(contentsOf: fileURL),
      let uiImage = UIImage(data: data)
    else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }

  @MainActor @available(iOS 16.0, *)
  public func callAsFunction(
    _ swiftUIImage: SwiftUI.Image,
    returnAnnotatedImage: Bool = true
  ) -> YOLOResult {
    let renderer = ImageRenderer(content: swiftUIImage)
    guard let uiImage = renderer.uiImage else {
      return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: [])
    }
    return self(uiImage, returnAnnotatedImage: returnAnnotatedImage)
  }
}
