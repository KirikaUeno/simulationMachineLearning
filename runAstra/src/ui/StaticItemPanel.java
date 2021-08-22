package ui;

import data.SimulationParameter;
import main.Config;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.ArrayList;

public class StaticItemPanel extends JPanel {
    private final MainPanel mainPanel;
    public JTable staticItemTable;
    public ArrayList<SimulationParameter> staticParameters = new ArrayList<>();

    public StaticItemPanel(MainPanel mainPanel){
        setPreferredSize(new Dimension(Config.boardWidth/2, Config.boardHeight));
        setFocusable(true);
        setName("staticItemPanel");
        this.mainPanel = mainPanel;
        initializeVariables();
    }

    private void initializeVariables(){
        staticItemTable = new JTable();
        DefaultTableModel dtm = new StaticTable(staticParameters);
        staticItemTable.setModel(dtm);
        dtm.addTableModelListener(staticItemTable);
        staticItemTable.setDefaultRenderer(Object.class, new ParameterTableRenderer());


        JScrollPane scrollPane = new JScrollPane(staticItemTable);

        SpringLayout layoutMain = new SpringLayout();
        layoutMain.putConstraint(SpringLayout.NORTH, scrollPane, 0, SpringLayout.NORTH, this);
        layoutMain.putConstraint(SpringLayout.SOUTH, scrollPane, 0, SpringLayout.SOUTH, this);
        layoutMain.putConstraint(SpringLayout.WEST, scrollPane, 0, SpringLayout.WEST, this);
        layoutMain.putConstraint(SpringLayout.EAST, scrollPane, 0, SpringLayout.EAST, this);
        setLayout(layoutMain);

        add(scrollPane);
    }
}
