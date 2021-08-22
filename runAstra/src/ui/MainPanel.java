package ui;

import data.InputParameter;
import data.OutputParameter;
import data.SimulationParameter;
import main.Config;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Random;

public class MainPanel extends JPanel {
    public final StaticItemPanel staticItemPanel = new StaticItemPanel(this);
    public final DynamicItemPanel dynamicItemPanel = new DynamicItemPanel(this);
    private final SpringLayout layout = new SpringLayout();
    private final TextField pathField = new TextField("run.in");
    private final Button loadPath = new Button("Load in file");
    private final Button sendRandomJob = new Button("Send random-sampled job");
    private final Button sendSteppedJob = new Button("Send stepped job");
    private final JLabel samplesNumber = new JLabel("samples number:");
    private final TextField samplesNumberField = new TextField("");
    private final JButton switchToML = new JButton("Make a prediction");
    private final JButton switchToSimulation = new JButton("Make a simulation dataset");
    private final Button moveLeft = new Button("<-");
    private final Button moveRight = new Button("->");
    public ArrayList<SimulationParameter> allParameters = new ArrayList<>();

    public final InputParametersPanel inputPanel = new InputParametersPanel(this);
    public final OutputParameterPanel outputPanel = new OutputParameterPanel(this);
    public final Button predictNN = new Button("predict NN");
    public final Button predictTrees = new Button("predict Tree");
    public final Button trainAndPredictNN = new Button("train and predict NN");
    public final Button trainAndPredictTrees = new Button("train and predict Tree");
    public final TextField pathMLField = new TextField("/");
    public final Button loadResults = new Button("load simulation results");

    private int fileNumber = 0;

    public MainPanel() {
        setPreferredSize(new Dimension(Config.boardWidth, Config.boardHeight));

        setFocusable(true);
        setName("mainPanel");
        initializeVariables();
        repaint();
    }

    private void initializeVariables(){
        pathField.setPreferredSize(new Dimension(200,25));
        samplesNumberField.setPreferredSize(new Dimension(40,20));
        loadPath.addActionListener(e -> loadList(pathField.getText()));
        sendSteppedJob.addActionListener(e -> {sendSteppedJob(0,new ArrayList<>());writeInputParameters();fileNumber=0;});
        sendRandomJob.addActionListener(e -> sendRandomJob());
        moveLeft.addActionListener(e->moveSelectedLeft());
        moveRight.addActionListener(e->moveSelectedRight());
        switchToML.addActionListener(e->switchToMLAction());
        switchToSimulation.addActionListener(e->switchToSimulationAction());
        loadResults.addActionListener(e->loadPath());
        trainAndPredictTrees.addActionListener(e->trainAndPredict());
        trainAndPredictNN.addActionListener(e->trainAndPredict());
        predictNN.addActionListener(e->predict());
        predictTrees.addActionListener(e->predict());

        layout.putConstraint(SpringLayout.WEST, pathField, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.NORTH, pathField, 10, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.WEST, loadPath, 15, SpringLayout.EAST, pathField);
        layout.putConstraint(SpringLayout.NORTH, loadPath, 10, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.EAST, sendRandomJob, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.SOUTH, sendRandomJob, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.EAST, samplesNumber, -5, SpringLayout.WEST, samplesNumberField);
        layout.putConstraint(SpringLayout.SOUTH, samplesNumber, -5, SpringLayout.NORTH, sendRandomJob);
        layout.putConstraint(SpringLayout.EAST, samplesNumberField, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.SOUTH, samplesNumberField, -5, SpringLayout.NORTH, sendRandomJob);
        layout.putConstraint(SpringLayout.EAST, sendSteppedJob, -5, SpringLayout.WEST, sendRandomJob);
        layout.putConstraint(SpringLayout.SOUTH, sendSteppedJob, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, moveLeft, 5, SpringLayout.WEST, staticItemPanel);
        layout.putConstraint(SpringLayout.NORTH, moveLeft, 5, SpringLayout.SOUTH, staticItemPanel);
        layout.putConstraint(SpringLayout.EAST, moveRight, -5, SpringLayout.EAST, dynamicItemPanel);
        layout.putConstraint(SpringLayout.NORTH, moveRight, 5, SpringLayout.SOUTH, dynamicItemPanel);
        layout.putConstraint(SpringLayout.EAST, switchToML, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, switchToML, 5, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.EAST, switchToSimulation, -5, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, switchToSimulation, 5, SpringLayout.NORTH, this);

        layout.putConstraint(SpringLayout.EAST, staticItemPanel, 0, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, staticItemPanel, 40, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.SOUTH, staticItemPanel, -80, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.EAST, dynamicItemPanel, 0, SpringLayout.WEST, staticItemPanel);
        layout.putConstraint(SpringLayout.NORTH, dynamicItemPanel, 40, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.SOUTH, dynamicItemPanel, -80, SpringLayout.SOUTH, staticItemPanel);
        layout.putConstraint(SpringLayout.WEST, dynamicItemPanel, 0, SpringLayout.WEST, this);

        layout.putConstraint(SpringLayout.EAST, outputPanel, 0, SpringLayout.EAST, this);
        layout.putConstraint(SpringLayout.NORTH, outputPanel, 40, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.SOUTH, outputPanel, -80, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.EAST, inputPanel, 0, SpringLayout.WEST, outputPanel);
        layout.putConstraint(SpringLayout.NORTH, inputPanel, 40, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.SOUTH, inputPanel, -80, SpringLayout.SOUTH, outputPanel);
        layout.putConstraint(SpringLayout.WEST, inputPanel, 0, SpringLayout.WEST, this);

        layout.putConstraint(SpringLayout.WEST, pathMLField, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.NORTH, pathMLField, 10, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.WEST, loadResults, 5, SpringLayout.EAST, pathMLField);
        layout.putConstraint(SpringLayout.NORTH, loadResults, 10, SpringLayout.NORTH, this);
        layout.putConstraint(SpringLayout.WEST, predictNN, 5, SpringLayout.WEST, this);
        layout.putConstraint(SpringLayout.SOUTH, predictNN, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, predictTrees, 5, SpringLayout.EAST, predictNN);
        layout.putConstraint(SpringLayout.SOUTH, predictTrees, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, trainAndPredictTrees, 5, SpringLayout.EAST, predictTrees);
        layout.putConstraint(SpringLayout.SOUTH, trainAndPredictTrees, -10, SpringLayout.SOUTH, this);
        layout.putConstraint(SpringLayout.WEST, trainAndPredictNN, 5, SpringLayout.EAST, trainAndPredictTrees);
        layout.putConstraint(SpringLayout.SOUTH, trainAndPredictNN, -10, SpringLayout.SOUTH, this);

        setLayout(layout);
        add(pathField);
        add(loadPath);
        add(sendRandomJob);
        add(sendSteppedJob);
        add(moveLeft);
        add(moveRight);
        add(dynamicItemPanel);
        add(staticItemPanel);
        add(samplesNumber);
        add(samplesNumberField);
        add(switchToML);
        add(switchToSimulation);
        add(inputPanel);
        add(outputPanel);
        add(trainAndPredictNN);
        add(trainAndPredictTrees);
        add(predictTrees);
        add(predictNN);
        add(pathMLField);
        add(loadResults);
        loadResults.setVisible(false);
        pathMLField.setVisible(false);
        predictNN.setVisible(false);
        predictTrees.setVisible(false);
        trainAndPredictTrees.setVisible(false);
        trainAndPredictNN.setVisible(false);
        outputPanel.setVisible(false);
        inputPanel.setVisible(false);
        switchToSimulation.setVisible(false);
        repaint();
    }

    public void sendRandomJob() {
        Random rand = new Random();
        for (int i = 0; i < Double.parseDouble(samplesNumberField.getText()); i++) {
            File original = new File("run.in");
            Path copied = Paths.get("run" + i + ".in");
            Path originalPath = original.toPath();
            try {
                Files.copy(originalPath, copied, StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                e.printStackTrace();
            }

            ArrayList<String> lines = new ArrayList<>();
            String line;
            try {
                File f1 = new File("run" + i + ".in");
                FileReader fr = new FileReader(f1);
                BufferedReader br = new BufferedReader(fr);
                while ((line = br.readLine()) != null) {
                    for(SimulationParameter p: dynamicItemPanel.dynamicParameters){
                        if (line.contains(p.getName())) {
                            if(p.getName().contains("MAXB")) line = p.getName() + "=" + (-((p.getUpperValue()-p.getLowerValue()) * rand.nextDouble() + p.getLowerValue())*0.000588-0.0000372);
                            else line = p.getName() + "=" + ((p.getUpperValue()-p.getLowerValue()) * rand.nextDouble() + p.getLowerValue());
                        }
                    }
                    lines.add(line);
                }
                fr.close();
                br.close();

                FileWriter fw = new FileWriter(f1);
                BufferedWriter out = new BufferedWriter(fw);
                for (String s : lines) {
                    out.write(s);
                    out.write("\n");
                }
                out.flush();
                fw.close();
                out.close();
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            sendBatch(i);
        }
        writeInputParameters();
        fileNumber = 0;
    }

    public void sendSteppedJob(int level, ArrayList<Double> values) {
        System.out.println("level: "+level+" out of " +(dynamicItemPanel.dynamicParameters.size()-1));
        System.out.println("fileNumber: "+fileNumber);
        double currentValue = dynamicItemPanel.dynamicParameters.get(level).getLowerValue();
        while (currentValue < dynamicItemPanel.dynamicParameters.get(level).getUpperValue()) {
            if ((level + 1) < dynamicItemPanel.dynamicParameters.size()) {
                System.out.println("opening new function");
                ArrayList<Double> valuesNew = new ArrayList<>(values);
                valuesNew.add(currentValue);
                sendSteppedJob(level + 1,valuesNew);
            } else {
                System.out.println("sending the job");
                File original = new File("run.in");
                Path copied = Paths.get("run" + fileNumber + ".in");
                Path originalPath = original.toPath();
                try {
                    Files.copy(originalPath, copied, StandardCopyOption.REPLACE_EXISTING);
                } catch (IOException e) {
                    e.printStackTrace();
                }

                ArrayList<String> lines = new ArrayList<>();
                String line;
                try {
                    File f1 = new File("run" + fileNumber + ".in");
                    FileReader fr = new FileReader(f1);
                    BufferedReader br = new BufferedReader(fr);
                    while ((line = br.readLine()) != null) {
                        for(int l = 0; (l+1) < dynamicItemPanel.dynamicParameters.size();l++){
                            SimulationParameter p = dynamicItemPanel.dynamicParameters.get(l);
                            if (line.contains(p.getName())) {
                                if(p.getName().contains("MAXB")) line = p.getName() + "=" + (-values.get(l)*0.000588-0.0000372);
                                else line = p.getName() + "=" + values.get(l);
                            }
                        }
                        SimulationParameter p = dynamicItemPanel.dynamicParameters.get(dynamicItemPanel.dynamicParameters.size()-1);
                        if (line.contains(p.getName())) {
                            if(p.getName().contains("MAXB")) line = p.getName() + "=" + (-currentValue*0.000588-0.0000372);
                            else line = p.getName() + "=" + currentValue;
                        }
                        lines.add(line);
                    }
                    fr.close();
                    br.close();

                    FileWriter fw = new FileWriter(f1);
                    BufferedWriter out = new BufferedWriter(fw);
                    for (String s : lines) {
                        out.write(s);
                        out.write("\n");
                    }
                    out.flush();
                    fw.close();
                    out.close();
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
                sendBatch(fileNumber);
                fileNumber++;
            }
            currentValue += dynamicItemPanel.dynamicParameters.get(level).getStep();
        }
    }

    public void sendBatch(int i){
        ArrayList<String> lines = new ArrayList<>();
        String line;
        try {
            File f1 = new File("batch.sh");
            FileReader fr = new FileReader(f1);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                if (line.contains("Astra")) {
                    line = "./Astra run" + i + ".in 2>&1 | tee run" + i + ".log";
                }
                if (line.contains("o_out")) {
                    line = "#$ -o o_out" + i + ".txt";
                }
                if (line.contains("e_out")) {
                    line = "#$ -e e_out" + i + ".txt";
                }
                lines.add(line);
            }
            fr.close();
            br.close();

            FileWriter fw = new FileWriter(f1);
            BufferedWriter out = new BufferedWriter(fw);
            for (String s : lines) {
                out.write(s);
                out.write("\n");
            }
            out.flush();
            fw.close();
            out.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        String s;
        Process p;
        try {
            p = Runtime.getRuntime().exec("qsub ./batch.sh");
            BufferedReader br = new BufferedReader(
                    new InputStreamReader(p.getInputStream()));
            while ((s = br.readLine()) != null)
                System.out.println("line: " + s);
            p.waitFor();
            System.out.println("exit: " + p.exitValue());
            p.destroy();
        } catch (Exception ignored) {
        }
    }

    public void writeInputParameters(){
        File f1 = new File("inputParameters.txt");
        FileWriter fw;
        try {
            fw = new FileWriter(f1);
            BufferedWriter out = new BufferedWriter(fw);
            for (SimulationParameter p : dynamicItemPanel.dynamicParameters) {
                out.write(p.getName());
                out.write("\n");
            }
            out.flush();
            fw.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadList(String path){
        String line;
        try {
            File f1 = new File(path);
            FileReader fr = new FileReader(f1);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                if (line.contains("=")) {
                    String[] parts = line.split("=");
                    if(isNumeric(parts[1])){
                        if(parts[0].contains("MAXB")) allParameters.add(new SimulationParameter(parts[0],(-Double.parseDouble(parts[1]) - 0.0000372) / 0.000588));
                        else allParameters.add(new SimulationParameter(parts[0],Double.parseDouble(parts[1])));
                    }
                }
            }
            fr.close();
            br.close();
            staticItemPanel.staticParameters = allParameters;
            DefaultTableModel dtm = new StaticTable(staticItemPanel.staticParameters);
            staticItemPanel.staticItemTable.setModel(dtm);
            dtm.addTableModelListener(staticItemPanel.staticItemTable);
            staticItemPanel.staticItemTable.setDefaultRenderer(Object.class, new ParameterTableRenderer());
            /*staticItemPanel.staticParameters = allParameters;
            ((DefaultTableModel) staticItemPanel.staticItemTable.getModel()).fireTableDataChanged();*/
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public boolean isNumeric(String strNum) {
        if (strNum == null) {
            return false;
        }
        try {
            double d = Double.parseDouble(strNum);
        } catch (NumberFormatException nfe) {
            return false;
        }
        return true;
    }

    public void moveSelectedLeft(){
        int[] rows = staticItemPanel.staticItemTable.getSelectedRows();
        for(int row:rows){
            staticItemPanel.staticParameters.get(row).isStatic = false;
        }
        updateStatus();
    }

    public void moveSelectedRight(){
        int[] rows = dynamicItemPanel.dynamicItemTable.getSelectedRows();
        for(int row:rows){
            dynamicItemPanel.dynamicParameters.get(row).isStatic = true;
        }
        updateStatus();
    }

    public void updateStatus(){
        dynamicItemPanel.dynamicParameters = new ArrayList<>();
        staticItemPanel.staticParameters = new ArrayList<>();
        for(SimulationParameter param: allParameters){
            if(param.isStatic) staticItemPanel.staticParameters.add(param);
            else dynamicItemPanel.dynamicParameters.add(param);
        }
        ((DynamicTable) dynamicItemPanel.dynamicItemTable.getModel()).auList = dynamicItemPanel.dynamicParameters;
        ((StaticTable) staticItemPanel.staticItemTable.getModel()).auList = staticItemPanel.staticParameters;
        ((DefaultTableModel) staticItemPanel.staticItemTable.getModel()).fireTableDataChanged();
        ((DefaultTableModel) dynamicItemPanel.dynamicItemTable.getModel()).fireTableDataChanged();
    }

    public void switchToMLAction(){
        loadResults.setVisible(true);
        pathMLField.setVisible(true);
        predictNN.setVisible(true);
        predictTrees.setVisible(true);
        trainAndPredictTrees.setVisible(true);
        trainAndPredictNN.setVisible(true);
        outputPanel.setVisible(true);
        inputPanel.setVisible(true);
        switchToSimulation.setVisible(true);
        pathField.setVisible(false);
        loadPath.setVisible(false);
        sendRandomJob.setVisible(false);
        sendSteppedJob.setVisible(false);
        moveLeft.setVisible(false);
        moveRight.setVisible(false);
        dynamicItemPanel.setVisible(false);
        staticItemPanel.setVisible(false);
        switchToML.setVisible(false);
        samplesNumber.setVisible(false);
        samplesNumberField.setVisible(false);
    }

    public void switchToSimulationAction(){
        loadResults.setVisible(false);
        pathMLField.setVisible(false);
        predictNN.setVisible(false);
        predictTrees.setVisible(false);
        trainAndPredictTrees.setVisible(false);
        trainAndPredictNN.setVisible(false);
        outputPanel.setVisible(false);
        inputPanel.setVisible(false);
        switchToSimulation.setVisible(false);
        pathField.setVisible(true);
        loadPath.setVisible(true);
        sendRandomJob.setVisible(true);
        sendSteppedJob.setVisible(true);
        moveLeft.setVisible(true);
        moveRight.setVisible(true);
        dynamicItemPanel.setVisible(true);
        staticItemPanel.setVisible(true);
        switchToML.setVisible(true);
        samplesNumber.setVisible(true);
        samplesNumberField.setVisible(true);
    }

    public void loadPath(){
        inputPanel.parameters.clear();
        File f1 = new File("inputParameters.txt");
        FileReader fr;
        String line;
        try {
            fr = new FileReader(f1);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                if(!line.equals("")){
                    inputPanel.parameters.add(new InputParameter(line));
                }
            }
            fr.close();
            br.close();
            ((DefaultTableModel) inputPanel.table.getModel()).fireTableDataChanged();
            System.out.println("changed input table:");
            System.out.println(inputPanel.parameters);
            Process process;
            process = Runtime.getRuntime().exec("/afs/ifh.de/user/k/kladov/volume/pythonLocal/bin/python3.9 getInformationDESY.py");
            InputStream stdout = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
            String line1;
            while ((line1 = reader.readLine()) != null) {
                System.out.println("stdout: " + line1);
            }
            //Runtime.getRuntime().exec("/afs/ifh.de/user/k/kladov/volume/pythonLocal/bin/python3.9 getInformationDESY.py");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void correctInformation(){
        File original = new File("information.txt");
        Path copied = Paths.get("informationCorrected.txt");
        Path originalPath = original.toPath();
        try {
            Files.copy(originalPath, copied, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            e.printStackTrace();
        }

        ArrayList<String> lines = new ArrayList<>();
        String line;
        try {
            File f1 = new File("informationCorrected.txt");
            FileReader fr = new FileReader(f1);
            BufferedReader br = new BufferedReader(fr);
            while ((line = br.readLine()) != null) {
                boolean isInList = true;
                for(OutputParameter p: outputPanel.parameters){
                    if(!p.estimate) {
                        if (line.contains(p.name)) {
                            isInList = false;
                            break;
                        }
                    }
                }
                if(isInList) lines.add(line);
            }
            fr.close();
            br.close();

            FileWriter fw = new FileWriter(f1);
            BufferedWriter out = new BufferedWriter(fw);
            for (String s : lines) {
                out.write(s);
                out.write("\n");
            }
            out.flush();
            fw.close();
            out.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public void saveInputParameters(){
        File f1 = new File("inputParametersValues.txt");
        FileWriter fw;
        try {
            fw = new FileWriter(f1);
            BufferedWriter out = new BufferedWriter(fw);
            for (InputParameter p: inputPanel.parameters) {
                if(p.name.contains("MAXB")) out.write(String.valueOf((-p.value*0.000588-0.0000372)));
                else out.write(String.valueOf(p.value));
                out.write("\n");
            }
            out.flush();
            fw.close();
            out.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void trainAndPredict(){
        System.out.println("start train");
        correctInformation();
        try {
            Process process;
            process = Runtime.getRuntime().exec("/afs/ifh.de/user/k/kladov/volume/pythonLocal/bin/python3.9 trainML.py");
            InputStream stdout = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
            String line1;
            while ((line1 = reader.readLine()) != null) {
                System.out.println("stdout: " + line1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        saveInputParameters();
        predict();
    }

    public void predict(){
        System.out.println("start predict");
        saveInputParameters();
        try {
            Process process;
            process = Runtime.getRuntime().exec("/afs/ifh.de/user/k/kladov/volume/pythonLocal/bin/python3.9 predict.py");
            InputStream stdout = process.getInputStream();
            BufferedReader reader = new BufferedReader(new InputStreamReader(stdout, StandardCharsets.UTF_8));
            String line1;
            while ((line1 = reader.readLine()) != null) {
                System.out.println("stdout: " + line1);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            ArrayList<String> outputNames = new ArrayList<>();
            File f = new File("informationCorrected.txt");
            FileReader fr = new FileReader(f);
            BufferedReader br = new BufferedReader(fr);
            String line;
            boolean isOutput = false;
            int numberOfInput = 0;
            while ((line = br.readLine()) != null) {
                if(line.contains("input") && numberOfInput==1){
                    break;
                } else if(line.contains("input")){
                    numberOfInput++;
                    isOutput = false;
                } else if (line.contains("output")){
                    isOutput = true;
                } else if (isOutput){
                    String[] parts = line.split(" ");
                    outputNames.add(parts[0]);
                }
            }
            fr.close();
            br.close();

            File f1 = new File("predictOutput.txt");
            fr = new FileReader(f1);
            br = new BufferedReader(fr);
            int i = 0;
            String name;
            while ((line = br.readLine()) != null) {
                name = outputNames.get(i);
                for(OutputParameter p: outputPanel.parameters){
                    if(name.equals(p.name)){
                        p.estimatedValue = Double.parseDouble(line);
                    }
                }
                i++;
            }
            fr.close();
            br.close();
            ((OutputParameterTable) outputPanel.table.getModel()).fireTableDataChanged();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }
}
