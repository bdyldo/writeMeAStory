import React from "react";

const Footer = () => (
  <div className="mt-12 text-center text-gray-400">
    <p className="text-sm">
      Powered by Self-Constructed Transformer model by{" "}
      <a
        href="https://github.com/bdyldo"
        className="underline text-white"
        target="_blank"
        rel="noopener noreferrer"
      >
        Dylan Sun
      </a>{" "}
      using PyTorch • Full-Stack Built with React + FastAPI + SocketIO
    </p>
  </div>
);

export default Footer;
