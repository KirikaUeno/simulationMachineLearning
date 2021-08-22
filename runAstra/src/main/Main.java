package main;

import ui.MainFrame;

import javax.swing.*;
import java.io.*;

public class Main {

    public static void main(String[] args) throws IOException {
        SwingUtilities.invokeLater(MainFrame::new);
    }
}
