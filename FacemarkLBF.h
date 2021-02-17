#ifndef FACEMARKLBF_H
#define FACEMARKLBF_H

#include "FacemarkLBFGlobal.h"
#include <opencv2/face.hpp>
#include "Core/CImageProcess2d.h"
#include "IO/CImageProcessIO.h"
#include "CPluginProcessInterface.hpp"

//------------------------------//
//----- CFacemarkLBFParam -----//
//------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFParam: public CProtocolTaskParam
{
    public:

        CFacemarkLBFParam() : CProtocolTaskParam()
        {
        }

        void setParamMap(const UMapString& paramMap) override
        {
            m_displayType = std::stoi(paramMap.at("displayType"));
        }

        UMapString  getParamMap() const override
        {
            UMapString map;
            map.insert(std::make_pair("displayType", std::to_string(m_displayType)));
            return map;
        }

    public:

        int m_displayType = 0;
};

//-------------------------//
//----- CFacemarkLBF -----//
//-------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBF : public CImageProcess2d
{
    public:

        CFacemarkLBF();
        CFacemarkLBF(const std::string name, const std::shared_ptr<CFacemarkLBFParam>& pParam);

        size_t  getProgressSteps() override;

        void    run() override;

    private:

        // drawPolyLine draws a poly line by joining successive points between the start and end indices.
        void    drawPolyline(std::shared_ptr<CGraphicsProcessOutput>& pOutput, std::vector<cv::Point2f> &landmarks, size_t start, size_t end, bool bIsClosed = false);
        void    drawLandmarksPoint(std::shared_ptr<CGraphicsProcessOutput>& pOutput, std::vector<cv::Point2f> &landmarks);
        void    drawLandmarksFace(std::shared_ptr<CGraphicsProcessOutput>& pOutput, std::vector<cv::Point2f> &landmarks);
        void    drawDelaunay(std::shared_ptr<CGraphicsProcessOutput>& pOutput, std::vector<cv::Point2f> &landmarks);

        void    manageInputGraphics(const CMat& imgSrc);
        void    manageOutput(std::vector<std::vector<cv::Point2f>>& landmarks);

    private:

        std::vector<cv::Rect> m_faces;
        cv::Ptr<cv::face::FacemarkLBF> m_pFacemark = nullptr;
};

//--------------------------------//
//----- CFacemarkLBFFactory -----//
//--------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFFactory : public CProcessFactory
{
    public:

        CFacemarkLBFFactory()
        {
            m_info.m_name = QObject::tr("Facemark LBF").toStdString();
            m_info.m_shortDescription = QObject::tr("Facial landmark detection using Local Binary Features (LBF)").toStdString();
            m_info.m_description = QObject::tr("The locations of the fiducial facial landmark points around facial components and "
                                               "facial contour capture the rigid and non-rigid facial deformations due to head movements and facial expressions. "
                                               "They are hence important for various facial analysis tasks. ").toStdString();
            m_info.m_path = QObject::tr("Plugins/C++/Face/Landmarks").toStdString();
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_keywords = "face,facial,landmark";
            m_info.m_authors = "Ren S, Cao X, Wei Y, Sun J.";
            m_info.m_article = "Face alignment at 3000 fps via regressing local binary features";
            m_info.m_journal = "CVPR";
            m_info.m_year = 2014;
            m_info.m_docLink = "https://docs.opencv.org/3.4.3/dc/d63/classcv_1_1face_1_1FacemarkLBF.html";
            m_info.m_license = "3-clause BSD License";
            m_info.m_repo = "https://github.com/opencv/opencv";
            m_info.m_version = "1.0.0";
        }

        virtual ProtocolTaskPtr create(const ProtocolTaskParamPtr& pParam) override
        {
            auto pFacemarkLBFParam = std::dynamic_pointer_cast<CFacemarkLBFParam>(pParam);
            if(pFacemarkLBFParam != nullptr)
                return std::make_shared<CFacemarkLBF>(m_info.m_name, pFacemarkLBFParam);
            else
                return create();
        }
        virtual ProtocolTaskPtr create() override
        {
            auto pFacemarkLBFParam = std::make_shared<CFacemarkLBFParam>();
            assert(pFacemarkLBFParam != nullptr);
            return std::make_shared<CFacemarkLBF>(m_info.m_name, pFacemarkLBFParam);
        }
};

//-------------------------------//
//----- CFacemarkLBFWidget -----//
//-------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFWidget: public CProtocolTaskWidget
{
    public:

        CFacemarkLBFWidget(QWidget *parent = Q_NULLPTR): CProtocolTaskWidget(parent)
        {
            init();
        }
        CFacemarkLBFWidget(ProtocolTaskParamPtr pParam, QWidget *parent = Q_NULLPTR): CProtocolTaskWidget(parent)
        {
            m_pParam = std::dynamic_pointer_cast<CFacemarkLBFParam>(pParam);
            init();
        }

    private:

        void init() override
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CFacemarkLBFParam>();

            auto pCombo = addCombo(0, tr("Display type"));
            pCombo->addItem("Points", 0);
            pCombo->addItem("Face", 1);
            pCombo->addItem("Delaunay", 2);
            pCombo->setCurrentIndex(pCombo->findData(m_pParam->m_displayType));

            connect(m_pApplyBtn, &QPushButton::clicked, [=]
            {
                m_pParam->m_displayType = pCombo->currentData().toInt();
                emit doApplyProcess(m_pParam);
            });
        }

    private:

        std::shared_ptr<CFacemarkLBFParam>  m_pParam = nullptr;
};

//--------------------------------------//
//----- CFacemarkLBFWidgetFactory -----//
//--------------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFWidgetFactory : public CWidgetFactory
{
    public:

        CFacemarkLBFWidgetFactory()
        {
            m_name = QObject::tr("Facemark LBF").toStdString();
        }

        virtual ProtocolTaskWidgetPtr   create(ProtocolTaskParamPtr pParam)
        {
            return std::make_shared<CFacemarkLBFWidget>(pParam);
        }
};

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class FACEMARKLBFSHARED_EXPORT CFacemarkLBFInterface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CProcessFactory> getProcessFactory()
        {
            return std::make_shared<CFacemarkLBFFactory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CFacemarkLBFWidgetFactory>();
        }
};

#endif // FACEMARKLBF_H
