package ui;

import main.Config;

import javax.swing.*;
import java.awt.*;

public class MainFrame extends JFrame {
    public MainFrame() {
        initializeLayout();
    }

    private void initializeLayout() {
        setTitle(Config.title);
        ImageIcon icon = new ImageIcon("images/appIcon.jpg");
        setIconImage(icon.getImage());

        MainPanel mainPanel = new MainPanel();
        setLayout(new BorderLayout());
        add(mainPanel);
        pack();
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        setResizable(true);
        setVisible(true);
    }
}
