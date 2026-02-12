fn main() {
    // tell the Windows linker where to find missing symbols
    println!("cargo:rustc-link-lib=shell32");
    println!("cargo:rustc-link-lib=ole32");
}
