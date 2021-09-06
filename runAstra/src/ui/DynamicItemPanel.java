package ui;

import data.SimulationParameter;
import main.Config;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.util.ArrayList;

public class DynamicItemPanel extends JPanel {
    public JTable dynamicItemTable;
    public ArrayList<SimulationParameter> dynamicParameters = new ArrayList<>();

    public DynamicItemPanel(){
        setPreferredSize(new Dimension(Config.boardWidth*2/3, Config.boardHeight));
        setFocusable(true);
        setName("dynamicItemPanel");
        initializeVariables();
    }

    private void initializeVariables(){
        dynamicItemTable = new JTable();
        DefaultTableModel dtm = new DynamicTable(dynamicParameters);
        dynamicItemTable.setModel(dtm);
        dtm.addTableModelListener(dynamicItemTable);
        dynamicItemTable.setDefaultRenderer(Object.class, new ParameterTableRenderer());


        JScrollPane scrollPane = new JScrollPane(dynamicItemTable);

        SpringLayout layoutMain = new SpringLayout();
        layoutMain.putConstraint(SpringLayout.NORTH, scrollPane, 0, SpringLayout.NORTH, this);
        layoutMain.putConstraint(SpringLayout.SOUTH, scrollPane, 0, SpringLayout.SOUTH, this);
        layoutMain.putConstraint(SpringLayout.WEST, scrollPane, 0, SpringLayout.WEST, this);
        layoutMain.putConstraint(SpringLayout.EAST, scrollPane, 0, SpringLayout.EAST, this);
        setLayout(layoutMain);

        add(scrollPane);
    }
}
